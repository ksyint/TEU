import sys
import os
import torch
from torch.utils.data import DataLoader

from common.config_utils import get_config
from common.device_utils import get_device
from common.logging_utils import set_seed, setup_logger
from reward.dataset import RankPairDataset, ScoreDataset
from reward.collator import PairCollator, ScoreCollator
from reward.trainer import RewardTrainer


def main():
    cfg = get_config()
    device = get_device(cfg)
    logger = setup_logger("train_reward", os.path.join(cfg["logging"]["output_dir"], "train.log"))
    set_seed(cfg["training"]["seed"])
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    if cfg["training"].get("rank_pair", True):
        train_dataset = RankPairDataset(cfg, split="train")
        valid_dataset = RankPairDataset(cfg, split="valid")
        collator = PairCollator()
    else:
        train_dataset = ScoreDataset(cfg, split="train")
        valid_dataset = ScoreDataset(cfg, split="valid")
        collator = ScoreCollator()

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Valid samples: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    trainer = RewardTrainer(cfg, device)
    logger.info("Starting training...")
    trainer.train(train_dataloader, valid_dataloader)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
