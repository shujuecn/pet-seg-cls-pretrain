from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .base import BaseTrainer
from .utils import build_scheduler, nt_xent_loss, save_ckpt


class PretrainTrainer(BaseTrainer):
    """Trainer for SimCLR pretraining."""

    def __init__(
        self,
        model: torch.nn.Module,
        optim_cfg,
        scheduler_cfg,
        device: str,
        temperature: float = 0.2,
        amp: bool = True,
        run_dir: Path | None = None,
        writer: SummaryWriter | None = None,
        logger=None,
    ) -> None:
        super().__init__(device)
        self.model = model.to(device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=float(optim_cfg.lr), weight_decay=float(optim_cfg.weight_decay)
        )
        self.temperature = temperature
        self.grad_clip_norm = float(optim_cfg.grad_clip_norm)
        self.grad_accum_steps = int(optim_cfg.grad_accum_steps)

        self.amp = bool(amp) and (device == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.scheduler_cfg = scheduler_cfg
        self.run_dir = run_dir
        self.writer = writer
        self.logger = logger

    def fit(self, loader, epochs: int, ckpt_path: str) -> None:
        best = 1e9
        scheduler = build_scheduler(self.opt, self.scheduler_cfg, t_max=epochs)

        for ep in range(1, epochs + 1):
            self.model.train()
            losses: List[float] = []
            pbar = tqdm(loader, desc=f"[pretrain] {ep}/{epochs}", leave=False)
            self.opt.zero_grad(set_to_none=True)

            for step, (x1, x2) in enumerate(pbar, start=1):
                x1, x2 = self.to_device([x1, x2])
                with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                    z1 = self.model(x1)
                    z2 = self.model(x2)
                    loss = nt_xent_loss(z1, z2, temperature=self.temperature) / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if step % self.grad_accum_steps == 0 or step == len(loader):
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)

                losses.append(loss.item() * self.grad_accum_steps)
                pbar.set_postfix(loss=float(loss.item() * self.grad_accum_steps))

            avg = float(np.mean(losses)) if losses else 0.0
            if scheduler:
                if self.scheduler_cfg.type == "plateau":
                    scheduler.step(avg)
                else:
                    scheduler.step()

            if self.writer:
                self.writer.add_scalar("pretrain/loss", avg, ep)
                self.writer.add_scalar("pretrain/lr", self.opt.param_groups[0]["lr"], ep)

            if self.logger:
                self.logger.info("[pretrain] ep%d: loss=%.4f", ep, avg)

            if avg < best:
                best = avg
                save_ckpt(ckpt_path, self.model, extra={"best_loss": best})
        if self.logger:
            self.logger.info("[pretrain] best_loss=%.4f saved: %s", best, ckpt_path)
