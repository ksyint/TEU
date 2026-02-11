import sys
import os
import json

from common.config_utils import get_config
from common.device_utils import get_device
from common.logging_utils import setup_logger
from reward.evaluator import RewardEvaluator


def main():
    cfg = get_config()
    device = get_device(cfg)
    logger = setup_logger("evaluate_reward", os.path.join(cfg["logging"]["output_dir"], "eval.log"))

    evaluator = RewardEvaluator(cfg, device)

    checkpoint_path = cfg["inference"]["checkpoint"]
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg["logging"]["output_dir"], "best_model.pth")
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    evaluator.load_model(checkpoint_path)

    data_cfg = cfg["data"]
    test_path = os.path.join(data_cfg["data_base"], f"{data_cfg.get('test_dataset', 'test')}.json")
    logger.info(f"Evaluating on: {test_path}")

    accuracy = evaluator.evaluate_dataset(test_path, data_cfg["image_base"])
    logger.info(f"Test Pairwise Accuracy: {accuracy:.4f}")

    results = {"test_accuracy": accuracy}
    output_path = os.path.join(cfg["logging"]["output_dir"], "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
