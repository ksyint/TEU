import logging
import math
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available

from dpo.losses import compute_log_probs, get_dpo_loss_fn
from dpo.reference_model import create_reference_model
from dpo.utils import import_model_class_from_model_name_or_path, compute_time_ids
from dpo.validation import log_dpo_validation
from common.device_utils import get_weight_dtype, set_tf32

logger = logging.getLogger(__name__)


class DPOSDXLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        log_cfg = cfg["logging"]
        device_cfg = cfg["device"]

        logging_dir = os.path.join(log_cfg["output_dir"], log_cfg.get("logging_dir", "logs"))
        project_config = ProjectConfiguration(project_dir=log_cfg["output_dir"], logging_dir=logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            mixed_precision=device_cfg.get("mixed_precision", "no"),
            log_with=log_cfg.get("report_to", "tensorboard"),
            project_config=project_config,
        )

        if train_cfg.get("seed") is not None:
            set_seed(train_cfg["seed"])
        if self.accelerator.is_main_process:
            os.makedirs(log_cfg["output_dir"], exist_ok=True)

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_cfg["pretrained_model_name_or_path"], subfolder="scheduler",
        )
        self.vae = AutoencoderKL.from_pretrained(
            model_cfg["pretrained_model_name_or_path"], subfolder="vae",
            revision=model_cfg.get("revision"),
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_cfg["pretrained_model_name_or_path"], subfolder="unet",
            revision=model_cfg.get("revision"),
        )

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.weight_dtype = get_weight_dtype(device_cfg.get("mixed_precision", "no"))
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        self.ref_model = create_reference_model(
            model_cfg["pretrained_model_name_or_path"],
            model_cfg.get("revision"),
            self.accelerator.device,
            self.weight_dtype,
        )

        if model_cfg.get("use_lora", True):
            unet_lora_config = LoraConfig(
                r=model_cfg.get("lora_rank", 4),
                lora_alpha=model_cfg.get("lora_rank", 4),
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)

        if device_cfg.get("enable_xformers") and is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        if train_cfg.get("gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
        if train_cfg.get("allow_tf32"):
            set_tf32(True)

        if train_cfg.get("use_8bit_adam"):
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        lora_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        self.optimizer = optimizer_cls(
            lora_params,
            lr=train_cfg["learning_rate"],
            betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
            weight_decay=train_cfg["adam_weight_decay"],
            eps=train_cfg["adam_epsilon"],
        )

        self.beta_dpo = train_cfg.get("beta_dpo", 5000)
        self.dpo_loss_fn = get_dpo_loss_fn(train_cfg.get("loss_type", "sigmoid"))

    def setup_dataloader(self, train_dataset, collate_fn):
        train_cfg = self.cfg["training"]
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn,
            batch_size=train_cfg["train_batch_size"], num_workers=0,
        )
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / train_cfg["gradient_accumulation_steps"]
        )
        if train_cfg.get("max_train_steps") is None:
            train_cfg["max_train_steps"] = train_cfg["num_train_epochs"] * self.num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            train_cfg.get("lr_scheduler", "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=train_cfg.get("lr_warmup_steps", 0) * train_cfg["gradient_accumulation_steps"],
            num_training_steps=train_cfg["max_train_steps"] * train_cfg["gradient_accumulation_steps"],
        )
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler,
        )
        train_cfg["num_train_epochs"] = math.ceil(train_cfg["max_train_steps"] / self.num_update_steps_per_epoch)
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.cfg["logging"].get("run_name", "dpo-sdxl"))

    def _encode_images(self, pixel_values):
        pixel_values = pixel_values.to(self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def _compute_loss(self, batch):
        train_cfg = self.cfg["training"]
        preferred_latents = self._encode_images(batch["preferred_pixel_values"])
        rejected_latents = self._encode_images(batch["rejected_pixel_values"])

        noise_preferred = torch.randn_like(preferred_latents)
        noise_rejected = torch.randn_like(rejected_latents)
        bsz = preferred_latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=preferred_latents.device)

        noisy_preferred = self.noise_scheduler.add_noise(preferred_latents, noise_preferred, timesteps)
        noisy_rejected = self.noise_scheduler.add_noise(rejected_latents, noise_rejected, timesteps)

        add_time_ids_preferred = torch.cat([
            compute_time_ids(
                train_cfg["resolution"], s, c, self.accelerator.device, self.weight_dtype
            ) for s, c in zip(batch["preferred_original_sizes"], batch["preferred_crop_top_lefts"])
        ])
        add_time_ids_rejected = torch.cat([
            compute_time_ids(
                train_cfg["resolution"], s, c, self.accelerator.device, self.weight_dtype
            ) for s, c in zip(batch["rejected_original_sizes"], batch["rejected_crop_top_lefts"])
        ])

        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device) if batch["prompt_embeds"] is not None else None
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.accelerator.device) if batch["pooled_prompt_embeds"] is not None else None

        cond_preferred = {"time_ids": add_time_ids_preferred, "text_embeds": pooled_prompt_embeds}
        cond_rejected = {"time_ids": add_time_ids_rejected, "text_embeds": pooled_prompt_embeds}

        policy_pred_preferred = self.unet(
            noisy_preferred, timesteps, prompt_embeds,
            added_cond_kwargs=cond_preferred, return_dict=False,
        )[0]
        policy_pred_rejected = self.unet(
            noisy_rejected, timesteps, prompt_embeds,
            added_cond_kwargs=cond_rejected, return_dict=False,
        )[0]

        ref_pred_preferred = self.ref_model.predict(
            noisy_preferred, timesteps, prompt_embeds,
            added_cond_kwargs=cond_preferred,
        )
        ref_pred_rejected = self.ref_model.predict(
            noisy_rejected, timesteps, prompt_embeds,
            added_cond_kwargs=cond_rejected,
        )

        policy_preferred_logps = compute_log_probs(self.noise_scheduler, policy_pred_preferred, noise_preferred, timesteps)
        policy_rejected_logps = compute_log_probs(self.noise_scheduler, policy_pred_rejected, noise_rejected, timesteps)
        ref_preferred_logps = compute_log_probs(self.noise_scheduler, ref_pred_preferred, noise_preferred, timesteps)
        ref_rejected_logps = compute_log_probs(self.noise_scheduler, ref_pred_rejected, noise_rejected, timesteps)

        loss, pref_reward, rej_reward = self.dpo_loss_fn(
            policy_preferred_logps, policy_rejected_logps,
            ref_preferred_logps, ref_rejected_logps,
            self.beta_dpo,
        )
        return loss, pref_reward, rej_reward

    def train(self):
        cfg = self.cfg
        train_cfg = cfg["training"]
        val_cfg = cfg["validation"]
        log_cfg = cfg["logging"]

        global_step = 0
        first_epoch = 0

        if train_cfg.get("resume_from_checkpoint"):
            path = train_cfg["resume_from_checkpoint"]
            if path == "latest":
                dirs = [d for d in os.listdir(log_cfg["output_dir"]) if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if dirs else None
            if path is not None:
                self.accelerator.load_state(os.path.join(log_cfg["output_dir"], path))
                global_step = int(path.split("-")[1])
                first_epoch = global_step // self.num_update_steps_per_epoch

        progress_bar = tqdm(
            range(0, train_cfg["max_train_steps"]),
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, train_cfg["num_train_epochs"]):
            self.unet.train()
            train_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    loss, pref_reward, rej_reward = self._compute_loss(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), train_cfg["max_grad_norm"])
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    train_loss += loss.detach().item()
                    self.accelerator.log({
                        "train_loss": loss.detach().item(),
                        "preferred_reward": pref_reward.item(),
                        "rejected_reward": rej_reward.item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

                    if self.accelerator.is_main_process and global_step % val_cfg["checkpointing_steps"] == 0:
                        save_path = os.path.join(log_cfg["output_dir"], f"checkpoint-{global_step}")
                        self.accelerator.save_state(save_path)

                        if val_cfg.get("validation_prompts"):
                            log_dpo_validation(
                                cfg, self.accelerator, self.vae, self.unet,
                                self.weight_dtype, global_step,
                            )

                progress_bar.set_postfix(
                    loss=loss.detach().item(),
                    lr=self.lr_scheduler.get_last_lr()[0],
                )

                if global_step >= train_cfg["max_train_steps"]:
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unet_unwrapped = self.accelerator.unwrap_model(self.unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unet_unwrapped)
            )
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=log_cfg["output_dir"],
                unet_lora_layers=unet_lora_state_dict,
            )
        self.accelerator.end_training()
