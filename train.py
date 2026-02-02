import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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
    seg_num_classes: int = 2

    # 数据与训练参数
    data_dir: str = "./data"
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


def dice_iou_from_confusion(confusion: torch.Tensor) -> Tuple[float, float]:
    """由混淆矩阵计算平均 Dice / IoU（按类平均）。"""
    num_classes = confusion.size(0)
    dice_scores = []
    iou_scores = []
    for cls in range(num_classes):
        tp = confusion[cls, cls].item()
        fp = confusion[:, cls].sum().item() - tp
        fn = confusion[cls, :].sum().item() - tp
        denom_dice = (2 * tp + fp + fn)
        denom_iou = (tp + fp + fn)
        dice_scores.append((2 * tp) / denom_dice if denom_dice > 0 else 0.0)
        iou_scores.append(tp / denom_iou if denom_iou > 0 else 0.0)
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def update_seg_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """更新分割的混淆矩阵。"""
    num_classes = confusion.size(0)
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    for cls in range(num_classes):
        for cls_pred in range(num_classes):
            confusion[cls, cls_pred] += ((targets_flat == cls) & (preds_flat == cls_pred)).sum().item()
    return confusion


def train_one_epoch_seg(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate_seg(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            confusion = update_seg_confusion(confusion, preds.cpu(), masks.cpu())

    dice, iou = dice_iou_from_confusion(confusion)
    return running_loss / len(loader.dataset), dice, iou


def train_one_epoch_cls(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


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
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for true_label, pred_label in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[true_label, pred_label] += 1

    acc = correct / total if total > 0 else 0.0
    return running_loss / len(loader.dataset), acc, confusion


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)


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
    model = UNet(in_channels=3, num_classes=cfg.seg_num_classes, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "seg"))
    best_dice = -1.0
    checkpoint_path = Path(cfg.log_dir) / "seg" / "best_seg.pt"

    for epoch in range(cfg.epochs):
        logging.info("[Seg] Epoch %d/%d", epoch + 1, cfg.epochs)
        train_loss = train_one_epoch_seg(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = evaluate_seg(model, val_loader, criterion, device, cfg.seg_num_classes)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/dice", val_dice, epoch)
        writer.add_scalar("metric/iou", val_iou, epoch)

        logging.info("[Seg] train_loss=%.4f val_loss=%.4f dice=%.4f iou=%.4f", train_loss, val_loss, val_dice, val_iou)

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, checkpoint_path)
            logging.info("[Seg] New best model saved: %.4f", best_dice)

    writer.close()


def run_cls(cfg: TrainConfig) -> None:
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=cfg.data_dir,
        task="cls",
        seg_num_classes=cfg.seg_num_classes,
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
        train_loss = train_one_epoch_cls(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, confusion = evaluate_cls(model, val_loader, criterion, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/acc", val_acc, epoch)

        logging.info("[Cls] train_loss=%.4f val_loss=%.4f acc=%.4f", train_loss, val_loss, val_acc)
        logging.info("[Cls] confusion_matrix=\n%s", confusion)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, checkpoint_path)
            logging.info("[Cls] New best model saved: %.4f", best_acc)

    writer.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
