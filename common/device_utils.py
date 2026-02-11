import torch


def get_device(cfg):
    device_cfg = cfg.get("device", {})
    gpu_id = device_cfg.get("gpu_id", "0")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    return device


def get_weight_dtype(mixed_precision):
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype


def set_tf32(allow_tf32):
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return {}
    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        info[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
        }
    return info
