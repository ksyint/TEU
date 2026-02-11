import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, score_better, score_worse):
        diff = score_better - score_worse
        loss = F.relu(self.margin - diff)
        return loss.mean()


class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score_better, score_worse):
        logits = score_better - score_worse
        labels = torch.ones_like(logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss


class ListwiseRankingLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, labels):
        scores = scores / self.temperature
        log_probs = F.log_softmax(scores, dim=-1)
        sorted_indices = torch.argsort(labels, descending=True)
        loss = 0.0
        for i in range(len(sorted_indices)):
            loss -= log_probs[sorted_indices[i]]
        return loss / len(sorted_indices)


class MSERankLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_scores, target_scores):
        return F.mse_loss(predicted_scores.squeeze(), target_scores.float())


def get_loss_fn(loss_type="bradley_terry", **kwargs):
    if loss_type == "pairwise":
        return PairwiseRankingLoss(margin=kwargs.get("margin", 1.0))
    elif loss_type == "bradley_terry":
        return BradleyTerryLoss()
    elif loss_type == "listwise":
        return ListwiseRankingLoss(temperature=kwargs.get("temperature", 1.0))
    elif loss_type == "mse":
        return MSERankLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
