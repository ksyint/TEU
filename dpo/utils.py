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


def encode_prompt_sdxl(batch, text_encoders, tokenizers, caption_column, is_train=True):
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
                captions, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            prompt_embeds = text_encoder(
                text_inputs.input_ids.to(text_encoder.device),
                output_hidden_states=True, return_dict=False,
            )
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def compute_time_ids(resolution, original_size, crops_coords_top_left, device, weight_dtype):
    target_size = (resolution, resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(device, dtype=weight_dtype)
    return add_time_ids
