import torch
import torch.nn.functional as F


def generate_timestep_weights(num_timesteps, bias_strategy="none", bias_portion=0.25,
                              bias_begin=0, bias_end=1000, bias_multiplier=1.0):
    weights = torch.ones(num_timesteps)
    num_to_bias = int(bias_portion * num_timesteps)
    if bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif bias_strategy == "range":
        range_begin = max(0, bias_begin)
        range_end = min(num_timesteps, bias_end)
        bias_indices = slice(range_begin, range_end)
    elif bias_strategy == "none":
        return weights
    else:
        return weights
    weights[bias_indices] *= bias_multiplier
    weights /= weights.sum()
    return weights


def add_noise_offset(noise, noise_offset):
    if noise_offset > 0:
        noise += noise_offset * torch.randn(
            (noise.shape[0], noise.shape[1], 1, 1), device=noise.device
        )
    return noise


def get_noise(batch_size, channels, height, width, device, dtype=torch.float32):
    return torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)


def compute_predicted_original(noise_scheduler, noise_pred, timestep, latents, weight_dtype):
    pred_original_sample = noise_scheduler.step(
        noise_pred, timestep, latents
    ).pred_original_sample.to(weight_dtype)
    return pred_original_sample


def decode_latents(vae, latents, weight_dtype):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents.to(weight_dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image
