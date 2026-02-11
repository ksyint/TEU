import torch


class PairCollator:
    def __call__(self, batch):
        text_ids = torch.stack([item["text_ids"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])
        img_better = torch.stack([item["img_better"] for item in batch])
        img_worse = torch.stack([item["img_worse"] for item in batch])
        return {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "img_better": img_better,
            "img_worse": img_worse,
        }


class ScoreCollator:
    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        text_ids = torch.stack([item["text_ids"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])
        scores = torch.tensor([item["score"] for item in batch], dtype=torch.float32)
        return {
            "images": images,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "scores": scores,
        }
