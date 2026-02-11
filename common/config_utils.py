import os
import yaml
import argparse
from copy import deepcopy


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    return args


def merge_args_into_config(args, cfg):
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
        if "train_batch_size" in cfg["training"]:
            cfg["training"]["train_batch_size"] = args.batch_size
    if args.epochs is not None:
        if "epochs" in cfg["training"]:
            cfg["training"]["epochs"] = args.epochs
        if "num_train_epochs" in cfg["training"]:
            cfg["training"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        if "lr" in cfg["training"]:
            cfg["training"]["lr"] = args.lr
        if "learning_rate" in cfg["training"]:
            cfg["training"]["learning_rate"] = args.lr
    if args.gpu_id is not None:
        if "device" in cfg:
            cfg["device"]["gpu_id"] = args.gpu_id
    if args.output_dir is not None:
        cfg["logging"]["output_dir"] = args.output_dir
    if args.checkpoint is not None:
        if "inference" in cfg:
            cfg["inference"]["checkpoint"] = args.checkpoint
    if args.resume_from_checkpoint is not None:
        cfg["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    return cfg


def get_config():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args_into_config(args, cfg)
    return cfg


def flatten_config(cfg, parent_key="", sep="."):
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items[new_key] = v
    return dict(items)


def config_to_namespace(cfg):
    flat = {}
    for section_key, section_val in cfg.items():
        if isinstance(section_val, dict):
            for k, v in section_val.items():
                flat[k] = v
        else:
            flat[section_key] = section_val
    return argparse.Namespace(**flat)
