from __future__ import annotations

from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch

from .splits import list_pairs
from utils.repro import seed_worker


def build_seg_tf(image_size: int, train: bool) -> A.Compose:
    base = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    if train:
        aug = [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.9, 1.1),
                rotate=(-45, 45),
                p=0.5,
            ),
        ]
        return A.Compose(aug + base)
    return A.Compose(base)


def build_cls_tf(image_size: int, train: bool) -> A.Compose:
    base = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    if train:
        aug = [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
            A.RandomResizedCrop(
                (image_size, image_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.6
            ),
        ]
        return A.Compose(aug + base)
    return A.Compose(base)


def build_pretrain_tf(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                (image_size, image_size), scale=(0.6, 1.0), ratio=(0.8, 1.2), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def pil_rgb_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)


def pil_mask_to_np(mask: Image.Image) -> np.ndarray:
    return np.array(mask, dtype=np.uint8)


def map_mask_to_train(mask: np.ndarray) -> np.ndarray:
    """Map raw mask labels {1,2,3} -> train labels {0,1,2}."""
    mapped = np.zeros_like(mask, dtype=np.uint8)
    mapped[mask == 1] = 0
    mapped[mask == 2] = 1
    mapped[mask == 3] = 2
    return mapped


class SegFromList(Dataset):
    """Segmentation dataset with synchronized image/mask augmentations."""

    def __init__(self, items: List[Dict], tf: A.Compose):
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, object]:
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        mask = Image.open(it["mask"])

        img_np = pil_rgb_to_np(img)
        mask_np = map_mask_to_train(pil_mask_to_np(mask))

        aug = self.tf(image=img_np, mask=mask_np)
        x = aug["image"]
        m = aug["mask"].long()

        return {
            "image": x,
            "mask": m,
            "image_path": it["image"],
            "mask_path": it["mask"],
        }


class ClsFromList(Dataset):
    def __init__(self, items: List[Dict], tf: A.Compose):
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, object]:
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        img_np = pil_rgb_to_np(img)
        aug = self.tf(image=img_np)
        x = aug["image"]
        y = int(it["label"])
        return {"image": x, "label": y, "image_path": it["image"]}


class PretrainImages(Dataset):
    """Return two augmented views for SimCLR."""

    def __init__(self, image_paths: List[str], tf: A.Compose):
        self.paths = image_paths
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        img_np = pil_rgb_to_np(img)
        x1 = self.tf(image=img_np)["image"]
        x2 = self.tf(image=img_np)["image"]
        return x1, x2


def _loader_kwargs(num_workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: bool) -> Dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return kwargs


def build_seg_loaders(
    split_dict: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tf_train = build_seg_tf(image_size, train=True)
    tf_eval = build_seg_tf(image_size, train=False)

    train_ds = SegFromList(split_dict["train"], tf_train)
    val_ds = SegFromList(split_dict["val"], tf_eval)
    test_ds = SegFromList(split_dict["test"], tf_eval)

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **_loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
        ),
    )


def build_cls_loaders(
    split_dict: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tf_train = build_cls_tf(image_size, train=True)
    tf_eval = build_cls_tf(image_size, train=False)

    train_ds = ClsFromList(split_dict["train"], tf_train)
    val_ds = ClsFromList(split_dict["val"], tf_eval)
    test_ds = ClsFromList(split_dict["test"], tf_eval)

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **_loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            **_loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
        ),
    )


def build_pretrain_loader(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
) -> DataLoader:
    tf = build_pretrain_tf(image_size)
    imgs, _ = list_pairs(data_root, "pretrain")
    ds = PretrainImages([str(p) for p in imgs], tf)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
        **_loader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
    )
