import torch
import torch.nn.functional as F
from PIL import Image


class CLIPScorer:
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", device="cpu"):
        self.device = device
        from transformers import CLIPModel, CLIPProcessor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.to(device)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

    def _encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return F.normalize(image_features, dim=-1)

    def _encode_text(self, text):
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)

    def score(self, prompt, image):
        txt_features = self._encode_text(prompt).float()
        img_features = self._encode_image(image).float()
        reward = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        return reward.detach().cpu().numpy().item()

    def score_batch(self, prompt, images):
        txt_features = self._encode_text(prompt).float()
        all_img_features = []
        for img in images:
            features = self._encode_image(img)
            all_img_features.append(features)
        img_features = torch.cat(all_img_features, dim=0).float()
        txt_features = txt_features.expand(img_features.size(0), -1)
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1)
        return rewards.detach().cpu().numpy().tolist()

    def inference_rank(self, prompt, image_paths):
        txt_features = self._encode_text(prompt).float()
        img_set = []
        for path in image_paths:
            features = self._encode_image(path)
            img_set.append(features)
        img_features = torch.cat(img_set, 0).float()
        txt_expanded = txt_features.expand(img_features.size(0), -1)
        rewards = torch.sum(torch.mul(txt_expanded, img_features), dim=1)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()
