import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from reward.models.reward_model import ImageRewardModel
from reward.transforms import get_transform


class RewardEvaluator:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = ImageRewardModel(cfg).to(device)
        self.model.eval()
        self.preprocess = get_transform(cfg["model"]["image_size"])

    def load_model(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def score_single(self, prompt, image_path):
        pil_image = Image.open(image_path).convert("RGB")
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_input = self.model.blip.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=35, return_tensors="pt",
        )
        text_ids = text_input.input_ids.to(self.device)
        text_mask = text_input.attention_mask.to(self.device)
        with torch.no_grad():
            reward = self.model(image, text_ids, text_mask)
        return reward.item()

    def rank_images(self, prompt, image_paths):
        scores = []
        for img_path in image_paths:
            s = self.score_single(prompt, img_path)
            scores.append(s)
        scores_np = np.array(scores)
        ranking = np.argsort(-scores_np) + 1
        return ranking.tolist(), scores

    def evaluate_dataset(self, data_path, image_base):
        with open(data_path, "r") as f:
            data = json.load(f)

        total = 0
        correct = 0
        for item in tqdm(data, desc="Evaluating"):
            prompt = item["prompt"]
            generations = item["generations"]
            labels = item["ranking"]
            image_paths = [os.path.join(image_base, g) for g in generations]

            _, scores = self.rank_images(prompt, image_paths)

            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] < labels[j]:
                        total += 1
                        if scores[i] > scores[j]:
                            correct += 1
                    elif labels[i] > labels[j]:
                        total += 1
                        if scores[i] < scores[j]:
                            correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def benchmark(self, prompts_path, img_dir, model_names):
        with open(prompts_path, "r") as f:
            prompt_list = json.load(f)

        results = {}
        for model_name in model_names:
            model_img_dir = os.path.join(img_dir, model_name)
            scores_list = []
            for item in tqdm(prompt_list, desc=f"Benchmark {model_name}"):
                prompt = item["prompt"]
                prompt_id = item["id"]
                img_files = [
                    f for f in os.listdir(model_img_dir)
                    if f.startswith(str(prompt_id))
                ]
                img_paths = [os.path.join(model_img_dir, f) for f in sorted(img_files)]
                if not img_paths:
                    continue
                _, scores = self.rank_images(prompt, img_paths)
                scores_list.append(np.mean(scores))
            results[model_name] = float(np.mean(scores_list)) if scores_list else 0.0
        return results
