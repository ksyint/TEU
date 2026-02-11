import numpy as np
import torch
from dpo.pipeline import create_dpo_validation_pipeline, run_dpo_inference


def log_dpo_validation(cfg, accelerator, vae, unet, weight_dtype, global_step):
    model_cfg = cfg["model"]
    val_cfg = cfg["validation"]
    prompts = val_cfg.get("validation_prompts", [])
    if not prompts:
        return

    pipeline = create_dpo_validation_pipeline(
        model_cfg["pretrained_model_name_or_path"],
        vae,
        accelerator.unwrap_model(unet),
        model_cfg.get("revision"),
        weight_dtype,
    )

    images = run_dpo_inference(
        pipeline, prompts, accelerator.device,
        seed=cfg["training"].get("seed"),
        num_inference_steps=val_cfg.get("num_inference_steps", 50),
    )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb
            tracker.log({
                "validation": [
                    wandb.Image(image, caption=f"{i}: {prompts[i]}")
                    for i, image in enumerate(images)
                ]
            })

    del pipeline
    torch.cuda.empty_cache()
