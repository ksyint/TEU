import os
import json
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
from reward.transforms import get_transform


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


class RankPairDataset(Dataset):
    def __init__(self, cfg, split="train"):
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        self.image_base = data_cfg["image_base"]
        self.preprocess = get_transform(model_cfg["image_size"])
        self.tokenizer = init_tokenizer()
        self.max_text_length = data_cfg.get("max_text_length", 35)
        dataset_name = data_cfg.get(f"{split}_dataset", split)
        pair_store_base = data_cfg.get("pair_store_base", None)
        pair_store_path = None
        if pair_store_base:
            pair_store_path = os.path.join(pair_store_base, f"{dataset_name}.pth")

        if pair_store_path and os.path.exists(pair_store_path):
            self.data = torch.load(pair_store_path)
        else:
            json_path = os.path.join(data_cfg["data_base"], f"{dataset_name}.json")
            with open(json_path, "r") as f:
                raw_data = json.load(f)
            self.data = self._make_pairs(raw_data)

        self.batch_size = cfg["training"]["batch_size"]

    def _make_pairs(self, raw_data):
        data_items = []
        for item in tqdm(raw_data, desc="Building pairs"):
            img_set = []
            for gen in item["generations"]:
                img_path = os.path.join(self.image_base, gen)
                pil_image = Image.open(img_path)
                image = self.preprocess(pil_image)
                img_set.append(image)

            text_input = self.tokenizer(
                item["prompt"],
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )

            labels = item["ranking"]
            for id_l in range(len(labels)):
                for id_r in range(id_l + 1, len(labels)):
                    dict_item = {}
                    dict_item["text_ids"] = text_input.input_ids.squeeze(0)
                    dict_item["text_mask"] = text_input.attention_mask.squeeze(0)
                    if labels[id_l] < labels[id_r]:
                        dict_item["img_better"] = img_set[id_l]
                        dict_item["img_worse"] = img_set[id_r]
                    elif labels[id_l] > labels[id_r]:
                        dict_item["img_better"] = img_set[id_r]
                        dict_item["img_worse"] = img_set[id_l]
                    else:
                        continue
                    data_items.append(dict_item)
        return data_items

    def save_pairs(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.data, path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ScoreDataset(Dataset):
    def __init__(self, cfg, split="train"):
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        self.image_base = data_cfg["image_base"]
        self.preprocess = get_transform(model_cfg["image_size"])
        self.tokenizer = init_tokenizer()
        self.max_text_length = data_cfg.get("max_text_length", 35)
        dataset_name = data_cfg.get(f"{split}_dataset", split)
        json_path = os.path.join(data_cfg["data_base"], f"{dataset_name}.json")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        self.data = self._build(raw_data)

    def _build(self, raw_data):
        items = []
        for entry in tqdm(raw_data, desc="Building score dataset"):
            prompt = entry["prompt"]
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            scores = entry.get("scores", entry.get("ranking", []))
            for i, gen in enumerate(entry["generations"]):
                img_path = os.path.join(self.image_base, gen)
                pil_image = Image.open(img_path)
                image = self.preprocess(pil_image)
                score = float(scores[i]) if i < len(scores) else 0.0
                items.append({
                    "image": image,
                    "text_ids": text_input.input_ids.squeeze(0),
                    "text_mask": text_input.attention_mask.squeeze(0),
                    "score": score,
                })
        return items

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
