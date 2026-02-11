import random
import numpy as np
import torch
from transformers import PretrainedConfig


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, cache_dir, subfolder="text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def encode_prompt(batch, text_encoders, tokenizers, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]
    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def encode_prompt_rm(batch, caption_column, reward_model):
    prompt_batch = batch[caption_column]
    text_inputs = reward_model.blip.tokenizer(
        prompt_batch, padding="max_length", truncation=True,
        max_length=35, return_tensors="pt",
    )
    return {
        "rm_input_ids": text_inputs.input_ids,
        "rm_attention_mask": text_inputs.attention_mask,
    }


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return {"model_input": model_input.cpu()}


def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    alpha = sqrt_alphas_cumprod[timesteps].float()
    sigma = sqrt_one_minus_alphas_cumprod[timesteps].float()
    snr = (alpha / sigma) ** 2
    return snr
