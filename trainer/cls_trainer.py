from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .base import BaseTrainer
from .utils import build_scheduler, save_ckpt


class ClsTrainer(BaseTrainer):
    """Trainer for cat/dog classification."""

    def __init__(
        self,
        model: torch.nn.Module,
        optim_cfg,
        scheduler_cfg,
        device: str,
        amp: bool,
        run_dir: Path,
        writer: SummaryWriter | None = None,
        logger=None,
    ) -> None:
        super().__init__(device)
        self.model = model.to(device)
        self.crit = nn.CrossEntropyLoss()
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
        self.run_dir = run_dir
        self.writer = writer
        self.logger = logger

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        losses: List[float] = []
        correct = 0
        total = 0

        for batch in tqdm(loader, desc="[cls] eval", leave=False):
            batch = self.to_device(batch)
            x = batch["image"]
            y = batch["label"]
            with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                logits = self.model(x)
                loss = self.crit(logits, y)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            losses.append(loss.item())

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "acc": correct / max(1, total),
        }

    def fit(self, train_loader, val_loader, epochs: int, ckpt_path: str) -> None:
        best = -1.0
        scheduler = build_scheduler(self.opt, self.scheduler_cfg, t_max=epochs)

        for ep in range(1, epochs + 1):
            self.model.train()
            losses: List[float] = []
            pbar = tqdm(train_loader, desc=f"[cls] {ep}/{epochs}", leave=False)
            self.opt.zero_grad(set_to_none=True)

            for step, batch in enumerate(pbar, start=1):
                batch = self.to_device(batch)
                x = batch["image"]
                y = batch["label"]

                with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                    logits = self.model(x)
                    loss = self.crit(logits, y) / self.grad_accum_steps

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
                self.writer.add_scalar("cls/train_loss", float(np.mean(losses)), ep)
                self.writer.add_scalar("cls/val_loss", val["loss"], ep)
                self.writer.add_scalar("cls/val_acc", val["acc"], ep)
                self.writer.add_scalar("cls/lr", lr, ep)

            if self.logger:
                self.logger.info(
                    "[cls] ep%d: val_loss=%.4f acc=%.4f", ep, val["loss"], val["acc"]
                )

            if val["acc"] > best:
                best = val["acc"]
                save_ckpt(ckpt_path, self.model, extra={"best_acc": best})
        if self.logger:
            self.logger.info("[cls] best_acc=%.4f saved: %s", best, ckpt_path)

    @torch.no_grad()
    def test_and_save(
        self,
        loader,
        out_dir: Path,
        save_topk_errors: bool,
        topk_errors: int,
    ) -> Dict[str, float]:
        self.model.eval()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "cls_test.csv"
        errors_dir = out_dir / "cls_errors"
        if save_topk_errors:
            errors_dir.mkdir(parents=True, exist_ok=True)

        rows = ["path,gt,pred,prob_cat,prob_dog"]
        total = 0
        correct = 0
        conf = np.zeros((2, 2), dtype=int)
        error_samples = []

        for batch in tqdm(loader, desc="[cls] test", leave=False):
            batch = self.to_device(batch)
            x = batch["image"]
            y = batch["label"]
            paths = batch["image_path"]

            with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                logits = self.model(x)
            probs = torch.softmax(logits.float(), dim=1)
            pred = probs.argmax(1)

            for i in range(x.size(0)):
                gt = int(y[i].item())
                pd = int(pred[i].item())
                prob_cat = float(probs[i, 0].item())
                prob_dog = float(probs[i, 1].item())
                path = paths[i]
                rows.append(f"{path},{gt},{pd},{prob_cat:.6f},{prob_dog:.6f}")
                conf[gt, pd] += 1
                total += 1
                correct += int(gt == pd)
                if gt != pd:
                    error_samples.append((max(prob_cat, prob_dog), path, gt, pd))

        csv_path.write_text("\n".join(rows), encoding="utf-8")

        if save_topk_errors and error_samples:
            error_samples.sort(key=lambda x: x[0], reverse=True)
            report_lines = [
                "# Classification Errors",
                "",
                f"Accuracy: {correct / max(1, total):.4f}",
                "",
            ]
            report_lines.append("Confusion Matrix (rows=gt, cols=pred)")
            report_lines.append(str(conf))

            for idx, (_, path, gt, pd) in enumerate(
                error_samples[:topk_errors], start=1
            ):
                img = Image.open(path).convert("RGB")
                out_path = errors_dir / f"err_{idx}_gt{gt}_pred{pd}.png"
                img.save(out_path)
            (errors_dir / "report.txt").write_text(
                "\n".join(report_lines), encoding="utf-8"
            )

        return {
            "acc": correct / max(1, total),
            "confusion": conf.tolist(),
        }
