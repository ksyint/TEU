import logging
import math
import os
import random
import shutil

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from rlhf.utils import import_model_class_from_model_name_or_path
from rlhf.sdxl_utils import compute_time_ids, unwrap_model, prepare_unet_added_conditions
from rlhf.noise_utils import decode_latents
from rlhf.validation import log_validation
from common.device_utils import get_weight_dtype, set_tf32

logger = logging.getLogger(__name__)


class RLHFSDXLTrainer:
    def __init__(self, cfg, reward_model_wrapper):
        self.cfg = cfg
        self.reward_model = reward_model_wrapper
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
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            model_cfg["pretrained_model_name_or_path"], model_cfg.get("revision"),
            cfg["data"].get("cache_dir"), subfolder="text_encoder",
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            model_cfg["pretrained_model_name_or_path"], model_cfg.get("revision"),
            cfg["data"].get("cache_dir"), subfolder="text_encoder_2",
        )
        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            model_cfg["pretrained_model_name_or_path"], subfolder="text_encoder",
            revision=model_cfg.get("revision"),
        )
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            model_cfg["pretrained_model_name_or_path"], subfolder="text_encoder_2",
            revision=model_cfg.get("revision"),
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
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.weight_dtype = get_weight_dtype(device_cfg.get("mixed_precision", "no"))
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)

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

        self.reward_model.to(self.accelerator.device, dtype=self.weight_dtype)
        self.rm_preprocess = self.reward_model.get_rm_transform()

    def setup_dataloader(self, precomputed_dataset, collate_fn):
        train_cfg = self.cfg["training"]
        self.train_dataloader = torch.utils.data.DataLoader(
            precomputed_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=train_cfg["train_batch_size"],
            num_workers=0,
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
            self.accelerator.init_trackers(
                self.cfg["logging"].get("run_name", "rlhf-sdxl"),
            )

    def _pretrain_loss(self, batch):
        train_cfg = self.cfg["training"]
        model_input = batch["model_input"].to(self.accelerator.device)
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        add_time_ids = torch.cat([
            compute_time_ids(
                train_cfg["resolution"], s, c, self.accelerator.device, self.weight_dtype
            ) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
        ])
        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.accelerator.device)
        unet_added_conditions = prepare_unet_added_conditions(add_time_ids, pooled_prompt_embeds)

        model_pred = self.unet(
            noisy_model_input, timesteps, prompt_embeds,
            added_cond_kwargs=unet_added_conditions, return_dict=False,
        )[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            target = noise

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def _reward_loss(self, batch):
        train_cfg = self.cfg["training"]
        add_time_ids = torch.cat([
            compute_time_ids(
                train_cfg["resolution"], s, c, self.accelerator.device, self.weight_dtype
            ) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
        ])
        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.accelerator.device)
        unet_added_conditions = prepare_unet_added_conditions(add_time_ids, pooled_prompt_embeds)

        latents = torch.randn(
            (train_cfg["train_batch_size"], 4, 128, 128), device=self.accelerator.device,
        )

        self.noise_scheduler.set_timesteps(40, device=self.accelerator.device)
        timesteps = self.noise_scheduler.timesteps
        mid_timestep = random.randint(30, 39)

        for i, t in enumerate(timesteps[:mid_timestep]):
            with torch.no_grad():
                latent_model_input = self.noise_scheduler.scale_model_input(latents, t)
                noise_pred = self.unet(
                    latent_model_input, t, prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                ).sample
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        latent_model_input = self.noise_scheduler.scale_model_input(latents, timesteps[mid_timestep])
        noise_pred = self.unet(
            latent_model_input, timesteps[mid_timestep], prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
        ).sample
        pred_original = self.noise_scheduler.step(
            noise_pred, timesteps[mid_timestep], latents,
        ).pred_original_sample.to(self.weight_dtype)

        image = decode_latents(self.vae, pred_original, self.weight_dtype)
        image = self.rm_preprocess(image).to(self.accelerator.device)

        rewards = self.reward_model.score_grad(
            batch["rm_input_ids"], batch["rm_attention_mask"], image,
        )
        loss = F.relu(-rewards + 2)
        loss = loss.mean() * train_cfg.get("grad_scale", 1e-3)
        return loss

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
                logs = {}

                if train_cfg.get("apply_pre_loss"):
                    with self.accelerator.accumulate(self.unet):
                        pre_loss = self._pretrain_loss(batch)
                        self.accelerator.backward(pre_loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.unet.parameters(), train_cfg["max_grad_norm"])
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    logs["pre_loss"] = pre_loss.detach().item()
                    train_loss += pre_loss.detach().item()

                if train_cfg.get("apply_reward_loss"):
                    self.noise_scheduler.set_timesteps(40, device=self.accelerator.device)
                    with self.accelerator.accumulate(self.unet):
                        reward_loss = self._reward_loss(batch)
                        self.accelerator.backward(reward_loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.unet.parameters(), train_cfg["max_grad_norm"])
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    logs["reward_loss"] = reward_loss.detach().item()
                    train_loss += reward_loss.detach().item()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if self.accelerator.is_main_process and global_step % val_cfg["checkpointing_steps"] == 0:
                        save_path = os.path.join(log_cfg["output_dir"], f"checkpoint-{global_step}")
                        self.accelerator.save_state(save_path)

                        if val_cfg.get("validation_prompts"):
                            log_validation(cfg, self.accelerator, self.vae, self.unet, self.weight_dtype, global_step)

                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)

                if global_step >= train_cfg["max_train_steps"]:
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unet_unwrapped = unwrap_model(self.accelerator, self.unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unet_unwrapped)
            )
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=log_cfg["output_dir"],
                unet_lora_layers=unet_lora_state_dict,
            )
        self.accelerator.end_training()
