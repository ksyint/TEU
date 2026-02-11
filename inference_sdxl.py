import os
import json
import argparse

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/inference/")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_inference_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16, variant="fp16",
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    if args.lora_path is not None:
        pipeline.load_lora_weights(args.lora_path)

    pipeline = pipeline.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    prompt_list = []
    if args.prompts is not None:
        with open(args.prompts, "r") as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, str):
                prompt_list.append(item)
            elif isinstance(item, dict):
                prompt_list.append(item["prompt"])
    elif args.prompt is not None:
        prompt_list = [args.prompt]
    else:
        prompt_list = ["a beautiful landscape painting"]

    all_results = []
    for idx, prompt in enumerate(prompt_list):
        for img_idx in range(args.num_images):
            image = pipeline(
                prompt=prompt,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            filename = f"prompt_{idx:04d}_img_{img_idx:02d}.png"
            save_path = os.path.join(args.output_dir, filename)
            image.save(save_path)
            all_results.append({"prompt": prompt, "image": filename})

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
