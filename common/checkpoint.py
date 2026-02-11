import os
import shutil
import torch


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, lr_scheduler, epoch, step, path):
    makedir(os.path.dirname(path))
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(model, path, optimizer=None, lr_scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    return model, optimizer, lr_scheduler, epoch, step


def cleanup_checkpoints(output_dir, total_limit):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) >= total_limit:
        num_to_remove = len(checkpoints) - total_limit + 1
        for ckpt in checkpoints[:num_to_remove]:
            ckpt_path = os.path.join(output_dir, ckpt)
            if os.path.isdir(ckpt_path):
                shutil.rmtree(ckpt_path)
            elif os.path.isfile(ckpt_path):
                os.remove(ckpt_path)


def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, checkpoints[-1])
