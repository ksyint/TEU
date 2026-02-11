import torch
from diffusers.utils.torch_utils import is_compiled_module


def compute_time_ids(resolution, original_size, crops_coords_top_left, device, weight_dtype):
    target_size = (resolution, resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(device, dtype=weight_dtype)
    return add_time_ids


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_unet_lora_target_modules():
    return [
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
        "conv1", "conv2",
        "conv_shortcut",
        "downsamplers.0.conv", "upsamplers.0.conv",
        "time_emb_proj",
    ]


def prepare_latent_size(resolution):
    return resolution // 8


def prepare_unet_added_conditions(time_ids, pooled_prompt_embeds):
    return {
        "time_ids": time_ids,
        "text_embeds": pooled_prompt_embeds,
    }
