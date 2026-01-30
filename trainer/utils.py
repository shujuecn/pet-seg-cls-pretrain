import random

import torch
import torch.nn as nn
import os


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    # pred/target: [B,H,W] {0,1}
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    return float(dice.mean().item())


@torch.no_grad()
def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - inter
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, mask01: torch.Tensor):
        # logits: [B,1,H,W], mask01: [B,H,W] {0,1}
        mask = mask01.float().unsqueeze(1)
        bce = self.bce(logits, mask)
        prob = torch.sigmoid(logits)
        inter = (prob * mask).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """
    SimCLR NT-Xent
    z1,z2: [B,D]，已归一化
    """
    B = z1.size(0)

    # 强制 float32 做相似度计算，避免 AMP 数值问题
    z1 = z1.float()
    z2 = z2.float()

    z = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = (z @ z.t()) / temperature  # [2B,2B]

    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e4)  # FP16/FP32 都安全

    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # [2B]
    denom = torch.logsumexp(sim, dim=1)
    loss = -(pos - denom).mean()
    return loss


def save_ckpt(path: str, model, extra: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ckpt(path: str, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def print_kv(prefix: str, d: dict):
    msg = " ".join([f"{k}={v}" for k, v in d.items()])
    print(f"{prefix} {msg}")
