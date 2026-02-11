import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class AnnealingLR(_LRScheduler):
    DECAY_STYLES = ["linear", "cosine", "exponential", "constant", "inverse_square_root"]

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters,
                 decay_style=None, last_iter=-1, decay_ratio=0.5):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.decay_ratio = decay_ratio
        self.step(self.num_iters)

    def get_lr(self):
        if self.decay_style == "inverse_square_root":
            return self.start_lr * math.sqrt(self.warmup_iter) / math.sqrt(max(self.warmup_iter, self.num_iters))
        elif self.decay_style == "constant":
            return self.start_lr
        else:
            if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
                return float(self.start_lr) * self.num_iters / self.warmup_iter
            else:
                if self.decay_style == "linear":
                    ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                    return self.start_lr - self.start_lr * (1 - self.decay_ratio) * ratio
                elif self.decay_style == "cosine":
                    ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                    return self.start_lr * (
                        (math.cos(math.pi * ratio) + 1) / 2 * (1 - self.decay_ratio) + self.decay_ratio
                    )
                else:
                    return self.start_lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def state_dict(self):
        return {
            "warmup_iter": self.warmup_iter,
            "num_iters": self.num_iters,
            "decay_style": self.decay_style,
            "end_iter": self.end_iter,
            "decay_ratio": self.decay_ratio,
        }

    def load_state_dict(self, sd):
        self.warmup_iter = sd["warmup_iter"]
        self.num_iters = sd["num_iters"]
        self.end_iter = sd["end_iter"]
        self.decay_style = sd["decay_style"]
        if "decay_ratio" in sd:
            self.decay_ratio = sd["decay_ratio"]
        self.step(self.num_iters)


def get_lr_scheduler(optimizer, cfg):
    tc = cfg["training"]
    epochs = tc.get("epochs", 10)
    batch_size = tc.get("batch_size", 32)
    accum = tc.get("accumulation_steps", 1)
    num_iters = epochs * max(1, 1000 // batch_size) // accum
    num_iters = max(1, num_iters)
    warmup_iter = int(tc.get("warmup", 0.01) * num_iters)
    return AnnealingLR(
        optimizer,
        start_lr=tc["lr"],
        warmup_iter=warmup_iter,
        num_iters=num_iters - warmup_iter,
        decay_style=tc.get("lr_decay_style", "cosine"),
        last_iter=-1,
        decay_ratio=tc.get("lr_decay_ratio", 0.0),
    )
