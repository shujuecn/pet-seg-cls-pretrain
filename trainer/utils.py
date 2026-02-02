from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg, t_max: int | None = None
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if cfg.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max or cfg.t_max, eta_min=cfg.eta_min
        )
    if cfg.type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.plateau_factor, patience=cfg.plateau_patience
        )
    return None


def dice_from_confusion(conf: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tp = torch.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice


def iou_from_confusion(conf: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tp = torch.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = (tp + eps) / (tp + fp + fn + eps)
    return iou


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    mask = (target >= 0) & (target < num_classes)
    idx = num_classes * target[mask] + pred[mask]
    conf = torch.bincount(idx, minlength=num_classes**2).view(num_classes, num_classes)
    return conf


class SegLoss(nn.Module):
    """CrossEntropy + Dice loss for multiclass segmentation."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice = multiclass_dice_loss(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice


def multiclass_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1)
    num_classes = probs.shape[1]
    target_onehot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2)
    target_onehot = target_onehot.float()
    dims = (0, 2, 3)
    inter = (probs * target_onehot).sum(dim=dims)
    union = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """SimCLR NT-Xent loss with safe float32 math."""
    bsz = z1.size(0)
    z1 = z1.float()
    z2 = z2.float()

    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / temperature

    mask = torch.eye(2 * bsz, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e4)

    pos = torch.cat([torch.diag(sim, bsz), torch.diag(sim, -bsz)], dim=0)
    denom = torch.logsumexp(sim, dim=1)
    loss = -(pos - denom).mean()
    return loss


def save_ckpt(path: str, model: nn.Module, extra: Dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ckpt(path: str, model: nn.Module, map_location: str = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def print_kv(prefix: str, d: Dict) -> None:
    msg = " ".join([f"{k}={v}" for k, v in d.items()])
    print(f"{prefix} {msg}")
