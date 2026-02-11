import os
import json
import argparse
import glob

import torch
from PIL import Image

from common.config_utils import load_config
from common.device_utils import get_device
from reward.evaluator import RewardEvaluator


def parse_score_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="outputs/scores.json")
    return parser.parse_args()


def main():
    args = parse_score_args()
    cfg = load_config(args.config)
    device = get_device(cfg)

    evaluator = RewardEvaluator(cfg, device)
    evaluator.load_model(args.checkpoint)

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
    image_files = sorted(image_files)

    if args.prompts_file is not None:
        with open(args.prompts_file, "r") as f:
            prompts_data = json.load(f)
    elif args.prompt is not None:
        prompts_data = [{"prompt": args.prompt, "images": [os.path.basename(p) for p in image_files]}]
    else:
        prompts_data = [{"prompt": "", "images": [os.path.basename(p) for p in image_files]}]

    all_scores = []
    for entry in prompts_data:
        prompt = entry["prompt"]
        images = entry.get("images", [os.path.basename(p) for p in image_files])
        img_paths = [os.path.join(args.image_dir, img) for img in images]
        img_paths = [p for p in img_paths if os.path.exists(p)]

        if len(img_paths) == 0:
            continue

        scores = []
        for img_path in img_paths:
            score = evaluator.score_single(prompt, img_path)
            scores.append(score)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        result = {
            "prompt": prompt,
            "images": images,
            "scores": scores,
            "ranking": [ranked_indices.index(i) + 1 for i in range(len(scores))],
        }
        all_scores.append(result)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    for result in all_scores:
        print(f"Prompt: {result['prompt'][:80]}")
        for img, score, rank in zip(result["images"], result["scores"], result["ranking"]):
            print(f"  Rank {rank}: {img} -> {score:.4f}")


if __name__ == "__main__":
    main()
