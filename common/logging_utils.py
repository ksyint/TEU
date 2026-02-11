import os
import sys
import logging
import random
import numpy as np
import torch


def setup_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    def __init__(self):
        self.metrics = {}

    def update(self, key, value, n=1):
        if key not in self.metrics:
            self.metrics[key] = AverageMeter()
        self.metrics[key].update(value, n)

    def get_avg(self, key):
        if key in self.metrics:
            return self.metrics[key].avg
        return 0.0

    def reset(self):
        for meter in self.metrics.values():
            meter.reset()

    def summary(self):
        return {k: v.avg for k, v in self.metrics.items()}
