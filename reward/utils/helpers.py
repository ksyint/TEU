import os
import random
import numpy as np
import torch
from huggingface_hub import hf_hub_download


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download_reward_model(version="ImageReward-v1.0", root=None):
    if root is None:
        root = os.path.expanduser("~/.cache/ImageReward")
    os.makedirs(root, exist_ok=True)
    filename = "ImageReward.pt"
    hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return os.path.join(root, filename)


def download_med_config(root=None):
    if root is None:
        root = os.path.expanduser("~/.cache/ImageReward")
    os.makedirs(root, exist_ok=True)
    filename = "med_config.json"
    hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return os.path.join(root, filename)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True
