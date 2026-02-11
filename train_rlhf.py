import sys
import os

from common.config_utils import load_config, parse_args, merge_args_into_config
from rlhf.reward_wrapper import RewardModelWrapper
from rlhf.trainer import RLHFSDXLTrainer
from rlhf.dataset import build_rlhf_dataset, build_collate_fn


def build_reward_config(cfg):
    reward_cfg = {
        "model": {
            "vit": "large",
            "image_size": 224,
            "med_config": os.path.join("configs", "med_config.json"),
            "mlp_hidden_dims": [1024, 128, 64, 16],
            "mlp_dropout": 0.2,
            "embed_dim": 256,
            "pretrained_path": cfg["model"]["image_reward_path"],
        },
    }
    return reward_cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args_into_config(args, cfg)

    reward_cfg = build_reward_config(cfg)
    reward_wrapper = RewardModelWrapper(reward_cfg, "cpu")

    trainer = RLHFSDXLTrainer(cfg, reward_wrapper)

    tokenizers = [
        trainer.text_encoder_one.tokenizer
        if hasattr(trainer.text_encoder_one, "tokenizer")
        else __import__("transformers").AutoTokenizer.from_pretrained(
            cfg["model"]["pretrained_model_name_or_path"], subfolder="tokenizer"
        ),
        __import__("transformers").AutoTokenizer.from_pretrained(
            cfg["model"]["pretrained_model_name_or_path"], subfolder="tokenizer_2"
        ),
    ]
    text_encoders = [trainer.text_encoder_one, trainer.text_encoder_two]

    precomputed_dataset = build_rlhf_dataset(
        cfg, tokenizers, text_encoders,
        trainer.vae, reward_wrapper.model, trainer.accelerator,
    )

    collate_fn = build_collate_fn(cfg)
    trainer.setup_dataloader(precomputed_dataset, collate_fn)
    trainer.train()


if __name__ == "__main__":
    main()
