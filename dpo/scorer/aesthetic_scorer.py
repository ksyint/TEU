import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AestheticMLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer:
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", model_path=None, device="cpu"):
        self.device = device
        from transformers import CLIPModel, CLIPProcessor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.to(device)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.mlp = AestheticMLP(768).to(device)
        if model_path is not None and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            self.mlp.load_state_dict(state_dict, strict=False)
        self.mlp.eval()

    def _get_image_features(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1).float()
        return image_features

    def score(self, prompt, image):
        image_features = self._get_image_features(image)
        with torch.no_grad():
            reward = self.mlp(image_features)
        return reward.detach().cpu().numpy().item()

    def score_batch(self, images):
        all_features = []
        for img in images:
            features = self._get_image_features(img)
            all_features.append(features)
        features = torch.cat(all_features, dim=0)
        with torch.no_grad():
            rewards = self.mlp(features)
        return rewards.squeeze(-1).detach().cpu().numpy().tolist()

    def inference_rank(self, prompt, image_paths):
        scores = []
        for path in image_paths:
            s = self.score(prompt, path)
            scores.append(s)
        rewards = torch.tensor(scores)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        return indices.numpy().tolist(), scores
