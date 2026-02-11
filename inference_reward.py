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
    logger = setup_logger("inference_reward")

    evaluator = RewardEvaluator(cfg, device)

    checkpoint_path = cfg["inference"]["checkpoint"]
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg["logging"]["output_dir"], "best_model.pth")
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    evaluator.load_model(checkpoint_path)

    prompts_path = cfg["inference"].get("prompts_path")
    image_dir = cfg["inference"].get("image_dir")

    if prompts_path is not None and image_dir is not None:
        with open(prompts_path, "r") as f:
            prompts_data = json.load(f)

        all_results = []
        for item in prompts_data:
            prompt = item["prompt"]
            images = item.get("images", [])
            image_paths = [os.path.join(image_dir, img) for img in images]

            if len(image_paths) > 1:
                ranking, scores = evaluator.rank_images(prompt, image_paths)
                result = {
                    "prompt": prompt,
                    "images": images,
                    "scores": scores,
                    "ranking": ranking,
                }
            elif len(image_paths) == 1:
                score = evaluator.score_single(prompt, image_paths[0])
                result = {
                    "prompt": prompt,
                    "image": images[0],
                    "score": score,
                }
            else:
                continue

            all_results.append(result)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"  Scores: {result.get('scores', result.get('score'))}")

        output_path = os.path.join(cfg["logging"]["output_dir"], "inference_results.json")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    else:
        logger.info("No prompts_path or image_dir specified in config inference section.")


if __name__ == "__main__":
    main()
