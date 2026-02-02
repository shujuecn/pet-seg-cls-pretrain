import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset


@dataclass
class SampleItem:
    image_path: Path
    mask_path: Path
    label: int


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

    注意：这个映射要在 Dataset 中完成，便于课堂讲解。
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


class SimpleAugmenter:
    """教学用的轻量数据增强：Resize + 随机水平翻转 + 轻微亮度/对比度。"""

    def __init__(
        self,
        image_size: int,
        train: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ):
        self.image_size = image_size
        self.train = train
        self.mean = mean
        self.std = std

    def _resize(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)
        return image, mask

    def _horizontal_flip(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def _brightness_contrast(self, image: Image.Image) -> Image.Image:
        if not self.train:
            return image
        if random.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(
                1.0 + random.uniform(-0.1, 0.1)
            )
            image = ImageEnhance.Contrast(image).enhance(
                1.0 + random.uniform(-0.1, 0.1)
            )
        return image

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        image_np = np.asarray(image).astype(np.float32) / 255.0
        image_np = (image_np - np.array(self.mean, dtype=np.float32)) / np.array(
            self.std, dtype=np.float32
        )
        image_np = np.transpose(image_np, (2, 0, 1))
        return torch.from_numpy(image_np)

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._resize(image, mask)
        image, mask = self._horizontal_flip(image, mask)
        image = self._brightness_contrast(image)
        image_tensor = self._to_tensor(image)
        mask_tensor = torch.from_numpy(np.asarray(mask).astype(np.int64))
        return image_tensor, mask_tensor


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
        self.augmenter = SimpleAugmenter(
            image_size=image_size,
            train=train,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")

        image_tensor, mask_tensor = self.augmenter(image, mask)

        if self.task == "seg":
            mapped = map_mask(mask_tensor.numpy(), self.seg_num_classes)
            return image_tensor, torch.from_numpy(mapped).long()

        if self.task == "cls":
            return image_tensor, torch.tensor(sample.label, dtype=torch.long)

        raise ValueError("task 只支持 'seg' 或 'cls'")


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
    image_root = Path(data_dir) / "image"
    mask_root = Path(data_dir) / "mask"

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
