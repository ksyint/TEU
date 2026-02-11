import torch
from contextlib import nullcontext
from diffusers import StableDiffusionXLPipeline


def create_dpo_validation_pipeline(pretrained_model_name_or_path, vae, unet, revision, weight_dtype):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        revision=revision,
        torch_dtype=weight_dtype,
    )
    return pipeline


def run_dpo_inference(pipeline, prompts, device, seed=None, num_inference_steps=50, guidance_scale=7.5):
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None
    images = []
    for prompt in prompts:
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(device.type)
        with autocast_ctx:
            image = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        images.append(image)
    return images


def save_dpo_pipeline(pretrained_model_name_or_path, vae, unet, revision, output_dir):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        revision=revision,
    )
    pipeline.save_pretrained(output_dir)
    del pipeline
    torch.cuda.empty_cache()
