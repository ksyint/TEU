import torch
from torchvision import transforms

from reward.models.reward_model import ImageRewardModel


class RewardModelWrapper:
    def __init__(self, cfg, device):
        self.device = device
        self.model = ImageRewardModel(cfg)
        pretrained_path = cfg["model"].get("pretrained_path", None)
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_rm_transform(self):
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def score_grad(self, input_ids, attention_mask, image):
        return self.model.score_grad(input_ids, attention_mask, image)

    def encode_text(self, prompts):
        text_inputs = self.model.blip.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=35, return_tensors="pt",
        )
        return text_inputs.input_ids.to(self.device), text_inputs.attention_mask.to(self.device)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.model = self.model.to(device, dtype=dtype)
        else:
            self.model = self.model.to(device)
        return self
