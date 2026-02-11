import os
import torch
import torch.nn as nn
from PIL import Image

from reward.models.blip_encoder import BLIPEncoder
from reward.models.mlp import MLP
from reward.transforms import get_transform


class ImageRewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        self.blip = BLIPEncoder(
            med_config_path=model_cfg["med_config"],
            image_size=model_cfg["image_size"],
            vit=model_cfg["vit"],
            embed_dim=model_cfg["embed_dim"],
        )
        self.mlp = MLP(
            input_size=768,
            hidden_dims=model_cfg.get("mlp_hidden_dims", [1024, 128, 64, 16]),
            dropout=model_cfg.get("mlp_dropout", 0.2),
        )
        self.preprocess = get_transform(model_cfg["image_size"])
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def forward(self, image, input_ids, attention_mask):
        txt_features = self.blip(image, input_ids, attention_mask)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        return rewards

    def score_grad(self, prompt_ids, prompt_attention_mask, image):
        image_embeds = self.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        text_output = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :]
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        return rewards

    def score(self, prompt, image_input, device=None):
        if device is None:
            device = next(self.parameters()).device
        if isinstance(image_input, list):
            _, rewards = self.inference_rank(prompt, image_input, device)
            return rewards
        if isinstance(image_input, Image.Image):
            pil_image = image_input
        elif isinstance(image_input, str) and os.path.isfile(image_input):
            pil_image = Image.open(image_input)
        else:
            raise TypeError("Unsupported image parameter type.")
        image = self.preprocess(pil_image).unsqueeze(0).to(device)
        text_input = self.blip.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=35, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            reward = self.forward(image, text_input.input_ids, text_input.attention_mask)
        return reward.detach().cpu().numpy().item()

    def inference_rank(self, prompt, image_paths, device=None):
        if device is None:
            device = next(self.parameters()).device
        text_input = self.blip.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=35, return_tensors="pt",
        ).to(device)
        txt_set = []
        for path in image_paths:
            if isinstance(path, Image.Image):
                pil_image = path
            elif isinstance(path, str):
                pil_image = Image.open(path)
            else:
                raise TypeError("Unsupported image parameter type.")
            image = self.preprocess(pil_image).unsqueeze(0).to(device)
            image_embeds = self.blip.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])
        txt_features = torch.cat(txt_set, 0).float()
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()
