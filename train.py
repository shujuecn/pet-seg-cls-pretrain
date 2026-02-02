import logging
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataloaders
from models import EncoderClassifier, UNet


class TrainConfig(BaseModel):
    # task: "seg" 或 "cls"
    task: str = "seg"

    # 分割类别数开关：2 或 3（2 表示把 mask 里 1 和 3 合并成前景）
    seg_num_classes: int = 2

    # 数据与训练参数
    data_dir: str = "./data/seg_cls"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42

    # 数据划分
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # 日志与模型保存
    log_dir: str = "runs/teach_minimal"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Seg metrics: confusion -> dice/iou
# -----------------------------
def fast_confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    快速计算分割混淆矩阵（num_classes x num_classes）
    preds/targets: (H,W) 或 (N, H, W)，值域 [0..C-1]
    """
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()

    # 去掉越界（理论上不会，但教学代码稳一点）
    mask = (
        (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    )
    preds = preds[mask]
    targets = targets[mask]

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes).view(
        num_classes, num_classes
    )
    return cm


def dice_iou_from_confusion(confusion: torch.Tensor) -> Tuple[float, float]:
    """
    由混淆矩阵计算平均 Dice / IoU（按类平均 macro）。
    confusion: (C, C), 行=GT, 列=Pred
    """
    confusion = confusion.float()
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return float(dice.mean().item()), float(iou.mean().item())


# -----------------------------
# Train/Eval loops: Seg
# -----------------------------
def train_one_epoch_seg(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Train(seg)", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)  # (B,C,H,W)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_seg(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float, torch.Tensor]:
    model.eval()
    running_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for images, masks in tqdm(loader, desc="Val(seg)", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)  # (B,H,W)
        cm_batch = fast_confusion_matrix(preds.cpu(), masks.cpu(), num_classes)
        confusion += cm_batch

    dice, iou = dice_iou_from_confusion(confusion)
    return running_loss / len(loader.dataset), dice, iou, confusion


# -----------------------------
# Train/Eval loops: Cls
# -----------------------------
def train_one_epoch_cls(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train(cls)", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)  # (B,2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_cls(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = np.zeros((2, 2), dtype=np.int64)

    for images, labels in tqdm(loader, desc="Val(cls)", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 教学：混淆矩阵用 numpy 直观
        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()
        for t, p in zip(y_true, y_pred):
            confusion[int(t), int(p)] += 1

    acc = correct / total if total > 0 else 0.0
    return running_loss / len(loader.dataset), acc, confusion


# -----------------------------
# Save
# -----------------------------
def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)


# -----------------------------
# Runs
# -----------------------------
def run_seg(cfg: TrainConfig) -> None:
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=cfg.data_dir,
        task="seg",
        seg_num_classes=cfg.seg_num_classes,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=cfg.seg_num_classes, base_channels=32).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "seg"))
    best_dice = -1.0
    checkpoint_path = Path(cfg.log_dir) / "seg" / "best_seg.pt"

    for epoch in range(cfg.epochs):
        logging.info("[Seg] Epoch %d/%d", epoch + 1, cfg.epochs)

        train_loss = train_one_epoch_seg(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_dice, val_iou, cm = evaluate_seg(
            model, val_loader, criterion, device, cfg.seg_num_classes
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/dice", val_dice, epoch)
        writer.add_scalar("metric/iou", val_iou, epoch)

        logging.info(
            "[Seg] train_loss=%.4f val_loss=%.4f dice=%.4f iou=%.4f",
            train_loss,
            val_loss,
            val_dice,
            val_iou,
        )
        logging.info("[Seg] confusion_matrix=\n%s", cm.numpy())

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, checkpoint_path)
            logging.info("[Seg] New best model saved: %.4f", best_dice)

    writer.close()


def run_cls(cfg: TrainConfig) -> None:
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=cfg.data_dir,
        task="cls",
        seg_num_classes=cfg.seg_num_classes,  # cls 不用它，但 Dataset 接口统一
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderClassifier(in_channels=3, num_classes=2, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "cls"))
    best_acc = -1.0
    checkpoint_path = Path(cfg.log_dir) / "cls" / "best_cls.pt"

    for epoch in range(cfg.epochs):
        logging.info("[Cls] Epoch %d/%d", epoch + 1, cfg.epochs)

        train_loss = train_one_epoch_cls(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, confusion = evaluate_cls(
            model, val_loader, criterion, device
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/acc", val_acc, epoch)

        logging.info(
            "[Cls] train_loss=%.4f val_loss=%.4f acc=%.4f",
            train_loss,
            val_loss,
            val_acc,
        )
        logging.info("[Cls] confusion_matrix=\n%s", confusion)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, checkpoint_path)
            logging.info("[Cls] New best model saved: %.4f", best_acc)

    writer.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    cfg = TrainConfig()

    set_seed(cfg.seed)

    if cfg.task == "seg":
        run_seg(cfg)
    elif cfg.task == "cls":
        run_cls(cfg)
    else:
        raise ValueError("task 只支持 'seg' 或 'cls'")


if __name__ == "__main__":
    main()
