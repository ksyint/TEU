import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class HPSHead(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.fc(x)


class HPSScorer:
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", model_path=None, device="cpu"):
        self.device = device
        from transformers import CLIPModel, CLIPProcessor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.to(device)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.head = HPSHead(768).to(device)
        if model_path is not None and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            self.head.load_state_dict(state_dict, strict=False)
        self.head.eval()

    def _encode(self, prompt, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.clip_processor(text=prompt, images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        image_features = F.normalize(outputs.image_embeds, dim=-1).float()
        text_features = F.normalize(outputs.text_embeds, dim=-1).float()
        combined = image_features * text_features
        return combined

    def score(self, prompt, image):
        combined = self._encode(prompt, image)
        with torch.no_grad():
            reward = self.head(combined)
        return reward.detach().cpu().numpy().item()

    def score_batch(self, prompt, images):
        scores = []
        for img in images:
            s = self.score(prompt, img)
            scores.append(s)
        return scores

    def inference_rank(self, prompt, image_paths):
        scores = self.score_batch(prompt, image_paths)
        rewards = torch.tensor(scores)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        return indices.numpy().tolist(), scores
