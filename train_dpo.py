import sys
import os
import functools

from common.config_utils import load_config, parse_args, merge_args_into_config
from dpo.trainer import DPOSDXLTrainer
from dpo.dataset import build_dpo_dataset, build_dpo_collate_fn
from dpo.utils import encode_prompt_sdxl, import_model_class_from_model_name_or_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args_into_config(args, cfg)

    trainer = DPOSDXLTrainer(cfg)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        model_cfg["pretrained_model_name_or_path"],
        model_cfg.get("revision"),
        data_cfg.get("cache_dir"),
        subfolder="text_encoder",
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        model_cfg["pretrained_model_name_or_path"],
        model_cfg.get("revision"),
        data_cfg.get("cache_dir"),
        subfolder="text_encoder_2",
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        subfolder="text_encoder",
        revision=model_cfg.get("revision"),
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        subfolder="text_encoder_2",
        revision=model_cfg.get("revision"),
    )

    from transformers import AutoTokenizer
    tokenizer_one = AutoTokenizer.from_pretrained(
        model_cfg["pretrained_model_name_or_path"], subfolder="tokenizer",
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_cfg["pretrained_model_name_or_path"], subfolder="tokenizer_2",
    )

    text_encoder_one.to(trainer.accelerator.device, dtype=trainer.weight_dtype)
    text_encoder_two.to(trainer.accelerator.device, dtype=trainer.weight_dtype)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    train_dataset = build_dpo_dataset(cfg, trainer.accelerator)

    caption_column = data_cfg.get("caption_column", "text")
    compute_embeddings_fn = functools.partial(
        encode_prompt_sdxl,
        text_encoders=[text_encoder_one, text_encoder_two],
        tokenizers=[tokenizer_one, tokenizer_two],
        caption_column=caption_column,
    )

    with trainer.accelerator.main_process_first():
        from datasets.fingerprint import Hasher
        new_fp = Hasher.hash(cfg)
        train_dataset = train_dataset.map(
            compute_embeddings_fn, batched=True,
            batch_size=128, new_fingerprint=new_fp,
        )

    del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    collate_fn = build_dpo_collate_fn()
    trainer.setup_dataloader(train_dataset, collate_fn)
    trainer.train()


if __name__ == "__main__":
    main()
