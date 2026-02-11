import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from reward.models.reward_model import ImageRewardModel
from reward.losses import get_loss_fn
from reward.utils.lr_scheduler import get_lr_scheduler
from reward.utils.helpers import set_seed
from common.logging_utils import AverageMeter
from common.checkpoint import save_checkpoint, load_checkpoint, makedir


class RewardTrainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = ImageRewardModel(cfg).to(device)
        self.train_cfg = cfg["training"]
        self.log_cfg = cfg["logging"]
        self._setup_optimizer()
        self._setup_loss()
        self.global_step = 0
        self.best_acc = 0.0

    def _setup_optimizer(self):
        tc = self.train_cfg
        if tc.get("fix_base", False):
            for name, param in self.model.blip.named_parameters():
                fix_rate = tc.get("fix_rate", 0.0)
                if fix_rate > 0:
                    import random
                    if random.random() < fix_rate:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=tc["lr"],
            betas=(tc["adam_beta1"], tc["adam_beta2"]),
            eps=tc["adam_eps"],
        )
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.cfg)

    def _setup_loss(self):
        loss_type = self.train_cfg.get("loss_type", "bradley_terry")
        margin = self.train_cfg.get("loss_margin", 1.0)
        self.loss_fn = get_loss_fn(loss_type, margin=margin)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        accumulation_steps = self.train_cfg["accumulation_steps"]
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            text_ids = batch["text_ids"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            img_better = batch["img_better"].to(self.device)
            img_worse = batch["img_worse"].to(self.device)

            score_better = self.model(img_better, text_ids, text_mask)
            score_worse = self.model(img_worse, text_ids, text_mask)

            loss = self.loss_fn(score_better, score_worse)
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                max_norm = self.train_cfg.get("max_grad_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            acc = (score_better > score_worse).float().mean().item()
            loss_meter.update(loss.item() * accumulation_steps, text_ids.size(0))
            acc_meter.update(acc, text_ids.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

        return loss_meter.avg, acc_meter.avg

    def validate(self, dataloader):
        self.model.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                img_better = batch["img_better"].to(self.device)
                img_worse = batch["img_worse"].to(self.device)
                score_better = self.model(img_better, text_ids, text_mask)
                score_worse = self.model(img_worse, text_ids, text_mask)
                acc = (score_better > score_worse).float().mean().item()
                acc_meter.update(acc, text_ids.size(0))
        return acc_meter.avg

    def train(self, train_dataloader, valid_dataloader=None):
        set_seed(self.train_cfg["seed"])
        makedir(self.log_cfg["output_dir"])
        epochs = self.train_cfg["epochs"]

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch)
            print(f"Epoch {epoch} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            if valid_dataloader is not None and epoch % self.log_cfg.get("valid_per_epoch", 1) == 0:
                val_acc = self.validate(valid_dataloader)
                print(f"Epoch {epoch} - Val Acc: {val_acc:.4f}")

                if self.log_cfg.get("save_best", True) and val_acc > self.best_acc:
                    self.best_acc = val_acc
                    path = os.path.join(self.log_cfg["output_dir"], "best_model.pth")
                    save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        epoch, self.global_step, path,
                    )
                    print(f"Best model saved with acc: {val_acc:.4f}")

            ckpt_path = os.path.join(
                self.log_cfg["output_dir"], f"checkpoint-{epoch}.pth"
            )
            save_checkpoint(
                self.model, self.optimizer, self.lr_scheduler,
                epoch, self.global_step, ckpt_path,
            )

    def load(self, checkpoint_path):
        self.model, self.optimizer, self.lr_scheduler, _, _ = load_checkpoint(
            self.model, checkpoint_path, self.optimizer, self.lr_scheduler, self.device
        )
