import os
import functools
import gc
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
from datasets import load_dataset, concatenate_datasets

from rlhf.utils import encode_prompt, encode_prompt_rm, compute_vae_encodings


def build_rlhf_dataset(cfg, tokenizers, text_encoders, vae, reward_model, accelerator):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if data_cfg.get("dataset_name") is not None:
        dataset = load_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("dataset_config_name"),
            cache_dir=data_cfg.get("cache_dir"),
        )
    else:
        data_files = {"train": data_cfg["train_data_dir"]}
        dataset = load_dataset("json", data_files=data_files, cache_dir=data_cfg.get("cache_dir"))

    column_names = dataset["train"].column_names
    image_column = data_cfg.get("image_column", "image")
    caption_column = data_cfg.get("caption_column", "text")

    train_resize = transforms.Resize(train_cfg["resolution"], interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(train_cfg["resolution"]) if train_cfg.get("center_crop") else transforms.RandomCrop(train_cfg["resolution"])
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def preprocess_train(examples):
        images = [
            Image.open(os.path.join(data_cfg.get("image_base_dir", ""), im_file)).convert("RGB")
            for im_file in examples[image_column]
        ]
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if train_cfg.get("random_flip") and random.random() < 0.5:
                image = train_flip(image)
            if train_cfg.get("center_crop"):
                y1 = max(0, int(round((image.height - train_cfg["resolution"]) / 2.0)))
                x1 = max(0, int(round((image.width - train_cfg["resolution"]) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (train_cfg["resolution"], train_cfg["resolution"]))
                image = crop(image, y1, x1, h, w)
            crop_top_lefts.append((y1, x1))
            image = train_transforms(image)
            all_images.append(image)
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        return examples

    with accelerator.main_process_first():
        if data_cfg.get("max_train_samples") is not None:
            dataset["train"] = dataset["train"].shuffle(seed=train_cfg["seed"]).select(
                range(data_cfg["max_train_samples"])
            )
        train_dataset = dataset["train"].with_transform(preprocess_train)

    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        caption_column=caption_column,
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
    compute_rm_encodings_fn = functools.partial(
        encode_prompt_rm,
        caption_column=caption_column,
        reward_model=reward_model,
    )

    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher
        new_fp = Hasher.hash(cfg)
        new_fp_vae = Hasher.hash(cfg["model"]["pretrained_model_name_or_path"])
        new_fp_rm = Hasher.hash(cfg["model"].get("image_reward_path", ""))

        ds_embeds = train_dataset.map(
            compute_embeddings_fn, batched=True,
            batch_size=data_cfg.get("mapping_batch_size", 128),
            new_fingerprint=new_fp,
        )
        ds_vae = train_dataset.map(
            compute_vae_encodings_fn, batched=True,
            batch_size=data_cfg.get("mapping_batch_size", 128),
            new_fingerprint=new_fp_vae,
        )
        ds_rm = train_dataset.map(
            compute_rm_encodings_fn, batched=True,
            batch_size=data_cfg.get("mapping_batch_size", 128),
            new_fingerprint=new_fp_rm,
        )

        id_column = data_cfg.get("id_column", "id")
        remove_cols = [c for c in [image_column, caption_column, id_column] if c in ds_vae.column_names]
        remove_cols_rm = [c for c in [image_column, caption_column, id_column] if c in ds_rm.column_names]

        precomputed_dataset = concatenate_datasets(
            [ds_embeds, ds_vae.remove_columns(remove_cols), ds_rm.remove_columns(remove_cols_rm)],
            axis=1,
        )
        precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)

    del compute_vae_encodings_fn, compute_embeddings_fn
    del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    return precomputed_dataset


def build_collate_fn(cfg):
    id_column = cfg["data"].get("id_column", "id")

    def collate_fn(examples):
        model_input = torch.stack([torch.tensor(ex["model_input"]) for ex in examples])
        original_sizes = [ex["original_sizes"] for ex in examples]
        crop_top_lefts = [ex["crop_top_lefts"] for ex in examples]
        prompt_embeds = torch.stack([torch.tensor(ex["prompt_embeds"]) for ex in examples])
        pooled_prompt_embeds = torch.stack([torch.tensor(ex["pooled_prompt_embeds"]) for ex in examples])
        rm_input_ids = torch.stack([torch.tensor(ex["rm_input_ids"]) for ex in examples])
        rm_attention_mask = torch.stack([torch.tensor(ex["rm_attention_mask"]) for ex in examples])
        rm_input_ids = rm_input_ids.view(-1, rm_input_ids.shape[-1])
        rm_attention_mask = rm_attention_mask.view(-1, rm_attention_mask.shape[-1])
        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "rm_input_ids": rm_input_ids,
            "rm_attention_mask": rm_attention_mask,
        }

    return collate_fn
