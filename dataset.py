from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class SampleItem:
    image_path: Path
    mask_path: Path
    label: int


# -----------------------------
# Mask 映射：{1,2,3} -> 教学用 {0,1,2} 或 {0,1}
# -----------------------------
def map_mask(mask_np: np.ndarray, seg_num_classes: int) -> np.ndarray:
    """
    将原始 mask 的像素值 {1,2,3} 映射到教学用的类别。

    原始约定：2 是背景；1 和 3 是前景的不同部分。

    - seg_num_classes = 3:
        背景(2) -> 0
        前景(1) -> 1
        前景(3) -> 2
    - seg_num_classes = 2:
        背景(2) -> 0
        前景(1 或 3) -> 1
    """
    if seg_num_classes not in (2, 3):
        raise ValueError("seg_num_classes 只支持 2 或 3")

    mapped = np.zeros_like(mask_np, dtype=np.int64)

    if seg_num_classes == 3:
        mapped[mask_np == 2] = 0
        mapped[mask_np == 1] = 1
        mapped[mask_np == 3] = 2
    else:
        mapped[mask_np == 2] = 0
        mapped[(mask_np == 1) | (mask_np == 3)] = 1

    return mapped


# -----------------------------
# Albumentations Transform
# -----------------------------
def build_albu_transform(
    image_size: int,
    train: bool,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> A.Compose:
    """
    教学用轻量增强（够讲、别复杂）：
    - Resize（image: bilinear / mask: nearest）
    - (train) HorizontalFlip
    - (train) RandomBrightnessContrast
    - Normalize
    - ToTensorV2

    注意：
    - mask_interpolation=NEAREST，避免 mask 类别被插值污染
    - ToTensorV2 后，mask 在不同版本中可能是 numpy 或 torch，这里在 Dataset 里统一兜底处理
    """
    ops = [
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=1,  # cv2.INTER_LINEAR
            mask_interpolation=0,  # cv2.INTER_NEAREST
        )
    ]

    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3,
            ),
        ]

    ops += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=False),
    ]
    return A.Compose(ops)


# -----------------------------
# 收集样本
# -----------------------------
def _collect_samples(image_root: Path, mask_root: Path) -> List[SampleItem]:
    samples: List[SampleItem] = []
    class_to_label = {"cat": 0, "dog": 1}

    for class_name in ("cat", "dog"):
        image_dir = image_root / class_name
        mask_dir = mask_root / class_name
        if not image_dir.exists():
            continue

        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        for image_path in image_paths:
            stem = image_path.stem
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            samples.append(
                SampleItem(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=class_to_label[class_name],
                )
            )

    return samples


# -----------------------------
# Dataset
# -----------------------------
class PetDataset(Dataset):
    """
    一个 Dataset 同时支持分割与分类：
    - task = "seg"：返回 (image_tensor, mask_tensor)
    - task = "cls"：返回 (image_tensor, label)
    """

    def __init__(
        self,
        samples: List[SampleItem],
        task: str,
        seg_num_classes: int,
        image_size: int,
        train: bool,
    ) -> None:
        self.samples = samples
        self.task = task
        self.seg_num_classes = seg_num_classes
        self.tf = build_albu_transform(
            image_size=image_size,
            train=train,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

        if self.task not in ("seg", "cls"):
            raise ValueError("task 只支持 'seg' 或 'cls'")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _ensure_numpy_mask(mask_any) -> np.ndarray:
        """
        ToTensorV2 后，不同版本可能返回：
        - mask 为 np.ndarray
        - mask 为 torch.Tensor
        这里统一转换为 np.ndarray，保证 map_mask 可用。
        """
        if torch.is_tensor(mask_any):
            return mask_any.detach().cpu().numpy()
        return mask_any

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # 读图：albumentations 习惯用 np.uint8
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")

        image_np = np.asarray(image)  # (H, W, 3) uint8
        mask_np = np.asarray(mask)  # (H, W) uint8 / int

        out = self.tf(image=image_np, mask=mask_np)

        image_tensor: torch.Tensor = out["image"]  # (3, H, W) float32
        mask_aug_any = out["mask"]  # numpy 或 torch（版本差异）

        if self.task == "seg":
            mask_aug_np = self._ensure_numpy_mask(mask_aug_any).astype(np.int64)
            mapped = map_mask(mask_aug_np, self.seg_num_classes)
            mask_tensor = torch.from_numpy(mapped).long()
            return image_tensor, mask_tensor

        # task == "cls"
        return image_tensor, torch.tensor(sample.label, dtype=torch.long)


# -----------------------------
# DataLoaders
# -----------------------------
def build_dataloaders(
    data_dir: str,
    task: str,
    seg_num_classes: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    image_root = Path(data_dir) / "images"
    mask_root = Path(data_dir) / "masks"

    samples = _collect_samples(image_root, mask_root)
    if len(samples) == 0:
        raise RuntimeError("未找到样本，请检查 data 目录路径")

    rng = torch.Generator().manual_seed(seed)
    total = len(samples)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size

    train_samples, val_samples, test_samples = torch.utils.data.random_split(
        samples, [train_size, val_size, test_size], generator=rng
    )

    train_dataset = PetDataset(
        list(train_samples),
        task=task,
        seg_num_classes=seg_num_classes,
        image_size=image_size,
        train=True,
    )
    val_dataset = PetDataset(
        list(val_samples),
        task=task,
        seg_num_classes=seg_num_classes,
        image_size=image_size,
        train=False,
    )
    test_dataset = PetDataset(
        list(test_samples),
        task=task,
        seg_num_classes=seg_num_classes,
        image_size=image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# quick sanity check
# -----------------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir="data/seg_cls",
        task="cls",
        seg_num_classes=2,
        image_size=256,
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    for images, masks in train_loader:
        print("images:", images.shape, images.dtype)  # (B,3,H,W)
        print("masks :", masks.shape, masks.dtype)  # (B,H,W)
        print("mask unique values (first sample):", torch.unique(masks[0]).tolist())
        break
