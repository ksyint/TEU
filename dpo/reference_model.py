import copy
import torch
from diffusers import UNet2DConditionModel


class ReferenceUNet:
    def __init__(self, pretrained_model_name_or_path, revision=None):
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
        )
        self.unet.requires_grad_(False)
        self.unet.eval()

    def to(self, device, dtype=None):
        if dtype is not None:
            self.unet = self.unet.to(device, dtype=dtype)
        else:
            self.unet = self.unet.to(device)
        return self

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.unet(*args, **kwargs)

    def predict(self, noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=None):
        with torch.no_grad():
            model_pred = self.unet(
                noisy_latents, timesteps, prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        return model_pred


def create_reference_model(pretrained_model_name_or_path, revision=None, device="cuda", dtype=torch.float32):
    ref_model = ReferenceUNet(pretrained_model_name_or_path, revision)
    ref_model.to(device, dtype)
    return ref_model
