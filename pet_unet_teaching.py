# pet_unet_teaching.py
# 教学版：Oxford-IIIT Pet + (1) Unet分割 (2) Encoder分类 (3) Encoder预训练->分割迁移
# 依赖：torch, torchvision, pydantic, tqdm

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# -----------------------------
# 1) 配置（不使用 argparse）
# -----------------------------
class Config(BaseModel):
    data_root: str = Field(default="./data", description="数据集下载/存放路径")
    image_size: int = Field(default=256, description="输入尺寸，教学建议 256 或 224")
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42

    # train/val split on trainval
    val_ratio: float = 0.15

    # 任务开关
    run_seg_from_scratch: bool = True
    run_encoder_cls: bool = True
    run_pretrain_then_seg: bool = True

    # 训练超参
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # epochs
    epochs_seg: int = 10
    epochs_cls: int = 8
    epochs_seg_ft: int = 10

    # 迁移学习策略
    freeze_encoder_epochs: int = 2  # 先冻结 encoder 训练 decoder 的 epoch 数

    # U-Net结构
    base_channels: int = 32
    depth: int = 4  # downsampling 次数（越大越重）


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 教学可复现：但会稍慢
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# 2) 数据：同一套 idx，支持分类/分割
# -----------------------------
class PetSegDataset(Dataset):
    """
    从 torchvision OxfordIIITPet 读 segmentation (trimap)，并转成二分类 mask:
    trimap: {0: background, 1: pet, 2: border}  (实际标注里常见 1/2 表示前景与边界)
    教学默认：mask = (trimap > 0) -> {0,1}
    """

    def __init__(self, root: str, split: str, image_tf, mask_tf):
        self.ds = OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=True
        )
        self.image_tf = image_tf
        self.mask_tf = mask_tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, trimap = self.ds[idx]  # PIL, PIL
        img = self.image_tf(img)
        # mask 用 NEAREST resize，避免插值出奇怪标签
        trimap = self.mask_tf(trimap)  # [H,W] long
        mask = (trimap > 0).long()  # 二分类 {0,1}
        return img, mask


class PetClsDataset(Dataset):
    """分类：同样的图片，但 target_types='category'，标签 0..36"""

    def __init__(self, root: str, split: str, image_tf):
        self.ds = OxfordIIITPet(
            root=root, split=split, target_types="category", download=True
        )
        self.image_tf = image_tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, y = self.ds[idx]
        img = self.image_tf(img)
        return img, int(y)


def make_transforms(image_size: int):
    image_tf = T.Compose(
        [
            T.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            # 教学用：简单归一化即可；你也可改成 ImageNet mean/std
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mask_tf = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            # PIL->Tensor 会变成 [1,H,W] float；我们转成 long [H,W]
            T.PILToTensor(),  # uint8 [1,H,W]
            T.Lambda(lambda x: x.squeeze(0).long()),
        ]
    )
    return image_tf, mask_tf


def split_trainval_indices(
    n: int, val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_n = int(round(n * val_ratio))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    return train_idx, val_idx


# -----------------------------
# 3) 模型：共享 Encoder
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = ConvBNReLU(in_ch, out_ch)
        self.c2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.c1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.c2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 对齐（理论上同尺寸，但防止奇数尺寸）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        return x


class UNetEncoder(nn.Module):
    """
    输出：
    - bottleneck feature
    - skips: list[Tensor]，用于decoder拼接
    """

    def __init__(self, base_channels=32, depth=4, in_channels=3):
        super().__init__()
        self.depth = depth
        chs = [base_channels * (2**i) for i in range(depth)]
        self.stem = DownBlock(in_channels, chs[0])

        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(DownBlock(chs[i - 1], chs[i]))
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DownBlock(chs[-1], chs[-1] * 2)
        self.out_channels = chs[-1] * 2
        self.skip_channels = chs  # 每层skip的通道数

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for blk in self.downs:
            x = self.pool(x)
            x = blk(x)
            skips.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        return x, skips  # skips包含每个尺度（从浅到深）


class UNet(nn.Module):
    def __init__(self, num_classes=1, base_channels=32, depth=4, in_channels=3):
        super().__init__()
        self.encoder = UNetEncoder(
            base_channels=base_channels, depth=depth, in_channels=in_channels
        )

        # decoder：从 bottleneck 开始，上采样 depth 次，拼接对应 skip
        skip_chs = self.encoder.skip_channels  # len=depth
        bottleneck_ch = self.encoder.out_channels

        self.ups = nn.ModuleList()
        # 从深到浅
        cur_ch = bottleneck_ch
        for i in reversed(range(depth)):
            self.ups.append(
                UpBlock(in_ch=cur_ch, skip_ch=skip_chs[i], out_ch=skip_chs[i])
            )
            cur_ch = skip_chs[i]

        self.head = nn.Conv2d(cur_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        # skips: [s0, s1, ..., s(depth-1)]，深度越大越深
        for up, i in zip(self.ups, reversed(range(len(skips)))):
            x = up(x, skips[i])
        logits = self.head(x)
        return logits  # [B,1,H,W]


class EncoderClassifier(nn.Module):
    def __init__(self, num_classes: int = 37, base_channels=32, depth=4, in_channels=3):
        super().__init__()
        self.encoder = UNetEncoder(
            base_channels=base_channels, depth=depth, in_channels=in_channels
        )
        self.fc = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x):
        feat, _ = self.encoder(x)  # [B,C,h,w]
        feat = F.adaptive_avg_pool2d(feat, 1)  # [B,C,1,1]
        feat = feat.flatten(1)  # [B,C]
        logits = self.fc(feat)
        return logits


# -----------------------------
# 4) 指标 & Loss
# -----------------------------
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    pred: [B,H,W] {0,1}
    target: [B,H,W] {0,1}
    """
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


class DiceBCELoss(nn.Module):
    """
    logits -> BCEWithLogits + soft Dice
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, mask: torch.Tensor):
        """
        logits: [B,1,H,W]
        mask:   [B,H,W] long {0,1}
        """
        mask_f = mask.float().unsqueeze(1)  # [B,1,H,W]
        bce = self.bce(logits, mask_f)

        prob = torch.sigmoid(logits)
        # soft dice
        inter = (prob * mask_f).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + mask_f.sum(dim=(2, 3))
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


# -----------------------------
# 5) 训练循环
# -----------------------------
@torch.no_grad()
def eval_seg(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    dices, ious, losses = [], [], []
    crit = DiceBCELoss()
    for x, mask in loader:
        x = x.to(device)
        mask = mask.to(device)
        logits = model(x)
        loss = crit(logits, mask)
        pred = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        dices.append(dice_coeff(pred, mask).item())
        ious.append(iou_score(pred, mask).item())
        losses.append(loss.item())
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "dice": float(sum(dices) / max(1, len(dices))),
        "iou": float(sum(ious) / max(1, len(ious))),
    }


def train_seg(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    tag: str,
):
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    crit = DiceBCELoss()

    best_val = -1.0
    best_path = f"{tag}_best.pt"

    for epoch in range(1, cfg.epochs_seg + 1):
        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"[{tag}] seg epoch {epoch}/{cfg.epochs_seg}",
            leave=False,
        )
        for x, mask in pbar:
            x = x.to(cfg.device)
            mask = mask.to(cfg.device)

            logits = model(x)
            loss = crit(logits, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        val = eval_seg(model, val_loader, cfg.device)
        if val["dice"] > best_val:
            best_val = val["dice"]
            torch.save(model.state_dict(), best_path)

        print(
            f"[{tag}] epoch {epoch}: val_loss={val['loss']:.4f} val_dice={val['dice']:.4f} val_iou={val['iou']:.4f}  (best_dice={best_val:.4f})"
        )

    print(f"[{tag}] saved best to: {best_path}")
    return best_path


@torch.no_grad()
def eval_cls(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    losses = []
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        losses.append(loss.item())
    acc = correct / max(1, total)
    return {"loss": float(sum(losses) / max(1, len(losses))), "acc": float(acc)}


def train_cls(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    tag: str,
):
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = f"{tag}_best.pt"

    for epoch in range(1, cfg.epochs_cls + 1):
        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"[{tag}] cls epoch {epoch}/{cfg.epochs_cls}",
            leave=False,
        )
        for x, y in pbar:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        val = eval_cls(model, val_loader, cfg.device)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), best_path)

        print(
            f"[{tag}] epoch {epoch}: val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} (best_acc={best_acc:.4f})"
        )

    print(f"[{tag}] saved best to: {best_path}")
    return best_path


def freeze_module(m: nn.Module, freeze: bool):
    for p in m.parameters():
        p.requires_grad = not freeze


def finetune_seg_with_pretrained_encoder(
    unet: UNet,
    encoder_ckpt: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    tag: str,
):
    # 加载分类预训练 encoder
    enc_state = torch.load(encoder_ckpt, map_location="cpu")
    # enc_state 是 classifier 的整体 state_dict，需要取出 encoder 部分
    # 我们保存时是整个 classifier_best.pt，因此这里按 key 前缀提取
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in enc_state.items()
        if k.startswith("encoder.")
    }
    unet.encoder.load_state_dict(encoder_state, strict=True)

    unet = unet.to(cfg.device)

    crit = DiceBCELoss()

    # 两阶段：先冻结 encoder -> 只训 decoder/head；然后解冻全部
    # 1) 冻结阶段
    if cfg.freeze_encoder_epochs > 0:
        freeze_module(unet.encoder, True)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        best_val = -1.0
        best_path = f"{tag}_best.pt"

        for epoch in range(1, cfg.freeze_encoder_epochs + 1):
            unet.train()
            pbar = tqdm(
                train_loader,
                desc=f"[{tag}] ft(frozen) {epoch}/{cfg.freeze_encoder_epochs}",
                leave=False,
            )
            for x, mask in pbar:
                x = x.to(cfg.device)
                mask = mask.to(cfg.device)
                logits = unet(x)
                loss = crit(logits, mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=float(loss.item()))
            val = eval_seg(unet, val_loader, cfg.device)
            if val["dice"] > best_val:
                best_val = val["dice"]
                torch.save(unet.state_dict(), best_path)
            print(
                f"[{tag}] frozen epoch {epoch}: val_dice={val['dice']:.4f} val_iou={val['iou']:.4f} (best_dice={best_val:.4f})"
            )
    else:
        best_path = f"{tag}_best.pt"
        best_val = -1.0

    # 2) 解冻全训阶段
    freeze_module(unet.encoder, False)
    opt = torch.optim.AdamW(
        unet.parameters(), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay
    )  # fine-tune 适当小点

    remain_epochs = cfg.epochs_seg_ft
    for epoch in range(1, remain_epochs + 1):
        unet.train()
        pbar = tqdm(
            train_loader,
            desc=f"[{tag}] ft(unfrozen) {epoch}/{remain_epochs}",
            leave=False,
        )
        for x, mask in pbar:
            x = x.to(cfg.device)
            mask = mask.to(cfg.device)
            logits = unet(x)
            loss = crit(logits, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        val = eval_seg(unet, val_loader, cfg.device)
        # 继续更新 best
        if val["dice"] > best_val:
            best_val = val["dice"]
            torch.save(unet.state_dict(), best_path)

        print(
            f"[{tag}] unfrozen epoch {epoch}: val_dice={val['dice']:.4f} val_iou={val['iou']:.4f} (best_dice={best_val:.4f})"
        )

    print(f"[{tag}] saved best to: {best_path}")
    return best_path


# -----------------------------
# 6) main：三段演示串起来
# -----------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.data_root, exist_ok=True)

    image_tf, mask_tf = make_transforms(cfg.image_size)

    # 基础数据集（trainval/test）
    seg_trainval = PetSegDataset(
        cfg.data_root, split="trainval", image_tf=image_tf, mask_tf=mask_tf
    )
    seg_test = PetSegDataset(
        cfg.data_root, split="test", image_tf=image_tf, mask_tf=mask_tf
    )

    cls_trainval = PetClsDataset(cfg.data_root, split="trainval", image_tf=image_tf)
    cls_test = PetClsDataset(cfg.data_root, split="test", image_tf=image_tf)

    assert len(seg_trainval) == len(cls_trainval), "trainval 长度应一致"
    assert len(seg_test) == len(cls_test), "test 长度应一致"

    # 关键：同一套 train/val idx
    train_idx, val_idx = split_trainval_indices(
        len(seg_trainval), cfg.val_ratio, cfg.seed
    )

    seg_train = Subset(seg_trainval, train_idx)
    seg_val = Subset(seg_trainval, val_idx)

    cls_train = Subset(cls_trainval, train_idx)
    cls_val = Subset(cls_trainval, val_idx)

    seg_train_loader = DataLoader(
        seg_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    seg_val_loader = DataLoader(
        seg_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    cls_train_loader = DataLoader(
        cls_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    cls_val_loader = DataLoader(
        cls_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"Device: {cfg.device}")
    print(
        f"Train/Val sizes: seg={len(seg_train)}/{len(seg_val)} cls={len(cls_train)}/{len(cls_val)}"
    )

    # (1) 完整 U-Net 分割（从头）
    if cfg.run_seg_from_scratch:
        unet_scratch = UNet(
            num_classes=1, base_channels=cfg.base_channels, depth=cfg.depth
        )
        cfg_seg = cfg.model_copy()
        # 用 cfg.epochs_seg
        train_seg(
            unet_scratch,
            seg_train_loader,
            seg_val_loader,
            cfg_seg,
            tag="unet_seg_scratch",
        )

    # (2) Encoder 分类
    encoder_ckpt_path = None
    if cfg.run_encoder_cls:
        clf = EncoderClassifier(
            num_classes=37, base_channels=cfg.base_channels, depth=cfg.depth
        )
        encoder_ckpt_path = train_cls(
            clf, cls_train_loader, cls_val_loader, cfg, tag="encoder_cls"
        )

    # (3) Encoder 预训练 -> U-Net 分割
    if cfg.run_pretrain_then_seg:
        if encoder_ckpt_path is None:
            # 如果你关闭了 run_encoder_cls，也可以手动填入之前训练得到的 encoder_cls_best.pt
            encoder_ckpt_path = "encoder_cls_best.pt"
        unet_ft = UNet(num_classes=1, base_channels=cfg.base_channels, depth=cfg.depth)
        finetune_seg_with_pretrained_encoder(
            unet=unet_ft,
            encoder_ckpt=encoder_ckpt_path,
            train_loader=seg_train_loader,
            val_loader=seg_val_loader,
            cfg=cfg,
            tag="unet_seg_pretrained",
        )


if __name__ == "__main__":
    main()
