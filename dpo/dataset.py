import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
from datasets import load_dataset


def build_dpo_dataset(cfg, accelerator):
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

    caption_column = data_cfg.get("caption_column", "text")
    image_column_preferred = data_cfg.get("image_column_preferred", "image_preferred")
    image_column_rejected = data_cfg.get("image_column_rejected", "image_rejected")
    image_base_dir = data_cfg.get("image_base_dir", "")

    train_resize = transforms.Resize(train_cfg["resolution"], interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop_fn = transforms.CenterCrop(train_cfg["resolution"]) if train_cfg.get("center_crop") else transforms.RandomCrop(train_cfg["resolution"])
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def process_image(img_path):
        image = Image.open(os.path.join(image_base_dir, img_path)).convert("RGB")
        original_size = (image.height, image.width)
        image = train_resize(image)
        if train_cfg.get("random_flip") and random.random() < 0.5:
            image = train_flip(image)
        if train_cfg.get("center_crop"):
            y1 = max(0, int(round((image.height - train_cfg["resolution"]) / 2.0)))
            x1 = max(0, int(round((image.width - train_cfg["resolution"]) / 2.0)))
            image = train_crop_fn(image)
        else:
            y1, x1, h, w = train_crop_fn.get_params(image, (train_cfg["resolution"], train_cfg["resolution"]))
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        image = train_transforms(image)
        return image, original_size, crop_top_left

    def preprocess_train(examples):
        preferred_images = []
        rejected_images = []
        preferred_sizes = []
        rejected_sizes = []
        preferred_crops = []
        rejected_crops = []

        for pref_path, rej_path in zip(examples[image_column_preferred], examples[image_column_rejected]):
            pref_img, pref_size, pref_crop = process_image(pref_path)
            rej_img, rej_size, rej_crop = process_image(rej_path)
            preferred_images.append(pref_img)
            rejected_images.append(rej_img)
            preferred_sizes.append(pref_size)
            rejected_sizes.append(rej_size)
            preferred_crops.append(pref_crop)
            rejected_crops.append(rej_crop)

        examples["preferred_pixel_values"] = preferred_images
        examples["rejected_pixel_values"] = rejected_images
        examples["preferred_original_sizes"] = preferred_sizes
        examples["rejected_original_sizes"] = rejected_sizes
        examples["preferred_crop_top_lefts"] = preferred_crops
        examples["rejected_crop_top_lefts"] = rejected_crops
        return examples

    with accelerator.main_process_first():
        if data_cfg.get("max_train_samples") is not None:
            dataset["train"] = dataset["train"].shuffle(seed=train_cfg["seed"]).select(
                range(data_cfg["max_train_samples"])
            )
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def build_dpo_collate_fn():
    def collate_fn(examples):
        preferred_pixel_values = torch.stack([ex["preferred_pixel_values"] for ex in examples])
        rejected_pixel_values = torch.stack([ex["rejected_pixel_values"] for ex in examples])
        preferred_original_sizes = [ex["preferred_original_sizes"] for ex in examples]
        rejected_original_sizes = [ex["rejected_original_sizes"] for ex in examples]
        preferred_crop_top_lefts = [ex["preferred_crop_top_lefts"] for ex in examples]
        rejected_crop_top_lefts = [ex["rejected_crop_top_lefts"] for ex in examples]
        prompt_embeds = torch.stack([torch.tensor(ex["prompt_embeds"]) for ex in examples]) if "prompt_embeds" in examples[0] else None
        pooled_prompt_embeds = torch.stack([torch.tensor(ex["pooled_prompt_embeds"]) for ex in examples]) if "pooled_prompt_embeds" in examples[0] else None
        return {
            "preferred_pixel_values": preferred_pixel_values,
            "rejected_pixel_values": rejected_pixel_values,
            "preferred_original_sizes": preferred_original_sizes,
            "rejected_original_sizes": rejected_original_sizes,
            "preferred_crop_top_lefts": preferred_crop_top_lefts,
            "rejected_crop_top_lefts": rejected_crop_top_lefts,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
    return collate_fn
