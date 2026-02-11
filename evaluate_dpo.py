import os
import json
import argparse
import glob

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from common.config_utils import load_config
from dpo.scorer import create_scorer


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_dpo/")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_eval_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = args.base_model or cfg["model"]["pretrained_model_name_or_path"]
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16,
    )
    if args.lora_path is not None:
        pipeline.load_lora_weights(args.lora_path)
    elif os.path.exists(os.path.join(cfg["logging"]["output_dir"], "pytorch_lora_weights.safetensors")):
        pipeline.load_lora_weights(cfg["logging"]["output_dir"])
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    scorer = create_scorer(cfg, device=device)

    with open(args.prompts_file, "r") as f:
        prompts_data = json.load(f)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    all_results = []

    for idx, item in enumerate(prompts_data):
        prompt = item if isinstance(item, str) else item["prompt"]
        prompt_scores = []

        for img_idx in range(args.num_images):
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]
            filename = f"prompt_{idx:04d}_img_{img_idx:02d}.png"
            save_path = os.path.join(args.output_dir, filename)
            image.save(save_path)
            score = scorer.score(prompt, image)
            prompt_scores.append({"image": filename, "score": score})

        avg_score = sum(s["score"] for s in prompt_scores) / len(prompt_scores)
        result = {
            "prompt": prompt,
            "avg_score": avg_score,
            "images": prompt_scores,
        }
        all_results.append(result)
        print(f"[{idx}] {prompt[:60]}... avg={avg_score:.4f}")

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    overall_avg = sum(r["avg_score"] for r in all_results) / len(all_results)
    print(f"Overall average score: {overall_avg:.4f}")

    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
