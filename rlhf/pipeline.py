import torch
from contextlib import nullcontext
from diffusers import StableDiffusionXLPipeline


def create_validation_pipeline(pretrained_model_name_or_path, vae, unet, revision, weight_dtype, prediction_type=None):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        revision=revision,
        torch_dtype=weight_dtype,
    )
    if prediction_type is not None:
        scheduler_args = {"prediction_type": prediction_type}
        pipeline.scheduler = pipeline.scheduler.from_config(
            pipeline.scheduler.config, **scheduler_args
        )
    return pipeline


def run_validation(pipeline, prompts, device, seed=None, num_inference_steps=50):
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
                generator=generator,
            ).images[0]
        images.append(image)
    return images


def save_pipeline(pretrained_model_name_or_path, vae, text_encoder, unet, revision, output_dir):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        revision=revision,
    )
    pipeline.save_pretrained(output_dir)
    del pipeline
    torch.cuda.empty_cache()
