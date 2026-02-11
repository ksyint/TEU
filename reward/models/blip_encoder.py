import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig

from reward.models.vit import VisionTransformer
from reward.models.med import BertModel


def create_vit(vit_type, image_size, drop_path_rate=0.0):
    if vit_type == "base":
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size, patch_size=16, embed_dim=vision_width,
            depth=12, num_heads=12, drop_path_rate=drop_path_rate,
        )
    elif vit_type == "large":
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size, patch_size=16, embed_dim=vision_width,
            depth=24, num_heads=16, drop_path_rate=drop_path_rate or 0.1,
        )
    else:
        raise ValueError(f"Unsupported vit type: {vit_type}")
    return visual_encoder, vision_width


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


class BLIPEncoder(nn.Module):
    def __init__(self, med_config_path, image_size=224, vit="large", embed_dim=256):
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit, image_size)
        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config_path)
        encoder_config.encoder_width = vision_width
        if not hasattr(encoder_config, "cross_attention_freq"):
            encoder_config.cross_attention_freq = 2
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

    def forward(self, image, input_ids, attention_mask):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        text_output = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]
