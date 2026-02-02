from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader.pet_ds import map_mask_to_train
from utils.visualization import PALETTE, mask_to_color, overlay_mask, save_index_md

from .base import BaseTrainer
from .utils import (
    SegLoss,
    build_scheduler,
    confusion_matrix,
    dice_from_confusion,
    iou_from_confusion,
    save_ckpt,
)


class SegTrainer(BaseTrainer):
    """Trainer for 3-class segmentation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optim_cfg,
        loss_cfg,
        scheduler_cfg,
        device: str,
        amp: bool,
        num_classes: int,
        run_dir: Path,
        writer: SummaryWriter | None = None,
        logger=None,
    ) -> None:
        super().__init__(device)
        self.model = model.to(device)
        self.crit = SegLoss(loss_cfg.seg_ce_weight, loss_cfg.seg_dice_weight)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(optim_cfg.lr),
            weight_decay=float(optim_cfg.weight_decay),
        )
        self.grad_clip_norm = float(optim_cfg.grad_clip_norm)
        self.grad_accum_steps = int(optim_cfg.grad_accum_steps)

        self.amp = bool(amp) and (device == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.scheduler_cfg = scheduler_cfg
        self.num_classes = num_classes
        self.run_dir = run_dir
        self.writer = writer
        self.logger = logger

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        losses: List[float] = []
        conf = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        for batch in loader:
            batch = self.to_device(batch)
            x = batch["image"]
            m = batch["mask"]
            with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                logits = self.model(x)
                loss = self.crit(logits, m)
            pred = logits.argmax(1)
            conf += confusion_matrix(pred, m, self.num_classes)
            losses.append(loss.item())

        dice = dice_from_confusion(conf).cpu().numpy()
        iou = iou_from_confusion(conf).cpu().numpy()
        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "dice": float(np.mean(dice)),
            "iou": float(np.mean(iou)),
        }
        for idx in range(self.num_classes):
            metrics[f"dice_c{idx}"] = float(dice[idx])
            metrics[f"iou_c{idx}"] = float(iou[idx])
        return metrics

    def _log_images(
        self, batch, logits: torch.Tensor, step: int, num_samples: int
    ) -> None:
        if not self.writer:
            return
        images = batch["image"]
        masks = batch["mask"]
        preds = logits.argmax(1)
        images = images[:num_samples]
        masks = masks[:num_samples]
        preds = preds[:num_samples]
        images = (images * 0.5 + 0.5).clamp(0, 1)
        masks = masks.unsqueeze(1).repeat(1, 3, 1, 1) / 2.0
        preds = preds.unsqueeze(1).repeat(1, 3, 1, 1) / 2.0
        grid = torch.cat([images, masks, preds], dim=0)
        self.writer.add_images("seg_samples", grid, global_step=step)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        ckpt_path: str,
        log_images_every: int,
        num_visual_samples: int,
    ) -> None:
        best = -1.0
        scheduler = build_scheduler(self.opt, self.scheduler_cfg, t_max=epochs)

        for ep in range(1, epochs + 1):
            self.model.train()
            losses = []
            pbar = tqdm(train_loader, desc=f"[seg] {ep}/{epochs}", leave=False)
            self.opt.zero_grad(set_to_none=True)
            for step, batch in enumerate(pbar, start=1):
                batch = self.to_device(batch)
                x = batch["image"]
                m = batch["mask"]

                with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                    logits = self.model(x)
                    loss = self.crit(logits, m) / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if step % self.grad_accum_steps == 0 or step == len(train_loader):
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)

                losses.append(loss.item() * self.grad_accum_steps)
                pbar.set_postfix(loss=float(loss.item() * self.grad_accum_steps))

            val = self.evaluate(val_loader)
            if scheduler:
                if self.scheduler_cfg.type == "plateau":
                    scheduler.step(val["loss"])
                else:
                    scheduler.step()

            lr = self.opt.param_groups[0]["lr"]
            if self.writer:
                self.writer.add_scalar("seg/train_loss", float(np.mean(losses)), ep)
                self.writer.add_scalar("seg/val_loss", val["loss"], ep)
                self.writer.add_scalar("seg/val_dice", val["dice"], ep)
                self.writer.add_scalar("seg/val_iou", val["iou"], ep)
                self.writer.add_scalar("seg/lr", lr, ep)
                for idx in range(self.num_classes):
                    self.writer.add_scalar(
                        f"seg/val_dice_c{idx}", val[f"dice_c{idx}"], ep
                    )
                    self.writer.add_scalar(
                        f"seg/val_iou_c{idx}", val[f"iou_c{idx}"], ep
                    )

            if log_images_every > 0 and ep % log_images_every == 0:
                batch = next(iter(val_loader))
                batch = self.to_device(batch)
                with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                    logits = self.model(batch["image"])
                self._log_images(batch, logits, ep, num_visual_samples)

            if self.logger:
                self.logger.info(
                    "[seg] ep%d: val_loss=%.4f dice=%.4f iou=%.4f",
                    ep,
                    val["loss"],
                    val["dice"],
                    val["iou"],
                )
            if val["dice"] > best:
                best = val["dice"]
                save_ckpt(ckpt_path, self.model, extra={"best_dice": best})
        if self.logger:
            self.logger.info("[seg] best_dice=%.4f saved: %s", best, ckpt_path)

    @torch.no_grad()
    def test_and_save(
        self,
        loader,
        out_dir: Path,
        save_original_size: bool,
        pred_mask_values: str,
        save_overlay: bool,
    ) -> Dict[str, float]:
        self.model.eval()
        out_dir.mkdir(parents=True, exist_ok=True)
        img_dir = out_dir / "images"
        gt_dir = out_dir / "gt"
        pred_dir = out_dir / "pred"
        overlay_dir = out_dir / "overlay"
        raw_dir = out_dir / "pred_mask"
        for d in [img_dir, gt_dir, pred_dir, overlay_dir, raw_dir]:
            d.mkdir(parents=True, exist_ok=True)

        conf = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        losses = []
        rows: List[Tuple[str, str, str, str]] = []

        for batch in tqdm(loader, desc="[seg] test", leave=False):
            batch = self.to_device(batch)
            x = batch["image"]
            m = batch["mask"]
            paths = batch["image_path"]
            mask_paths = batch["mask_path"]

            with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                logits = self.model(x)
                loss = self.crit(logits, m)
            pred = logits.argmax(1)
            conf += confusion_matrix(pred, m, self.num_classes)
            losses.append(loss.item())

            for i in range(pred.shape[0]):
                image_path = Path(paths[i])
                mask_path = Path(mask_paths[i])
                stem = image_path.stem

                image = Image.open(image_path).convert("RGB")
                gt_mask = map_mask_to_train(np.array(Image.open(mask_path)))
                pred_mask = pred[i].cpu().numpy().astype(np.uint8)

                if save_original_size:
                    if pred_mask.shape[::-1] != image.size:
                        pred_mask = np.array(
                            Image.fromarray(pred_mask).resize(
                                image.size, resample=Image.NEAREST
                            )
                        )
                    if gt_mask.shape[::-1] != image.size:
                        gt_mask = np.array(
                            Image.fromarray(gt_mask).resize(
                                image.size, resample=Image.NEAREST
                            )
                        )

                image.save(img_dir / f"{stem}.png")
                mask_to_color(gt_mask, PALETTE).save(gt_dir / f"{stem}.png")
                mask_to_color(pred_mask, PALETTE).save(pred_dir / f"{stem}.png")

                if save_overlay:
                    overlay = overlay_mask(image, pred_mask)
                    overlay.save(overlay_dir / f"{stem}.png")
                else:
                    overlay = Image.new("RGB", image.size)
                    overlay.save(overlay_dir / f"{stem}.png")

                if pred_mask_values == "one_based":
                    raw = pred_mask + 1
                else:
                    raw = pred_mask
                Image.fromarray(raw.astype(np.uint8)).save(raw_dir / f"{stem}.png")

                rows.append(
                    (
                        f"images/{stem}.png",
                        f"gt/{stem}.png",
                        f"pred/{stem}.png",
                        f"overlay/{stem}.png",
                    )
                )

        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "dice": float(np.mean(dice_from_confusion(conf).cpu().numpy())),
            "iou": float(np.mean(iou_from_confusion(conf).cpu().numpy())),
        }
        for idx in range(self.num_classes):
            metrics[f"dice_c{idx}"] = float(
                dice_from_confusion(conf).cpu().numpy()[idx]
            )
            metrics[f"iou_c{idx}"] = float(iou_from_confusion(conf).cpu().numpy()[idx])

        save_index_md(rows, out_dir / "summary.md")
        return metrics
