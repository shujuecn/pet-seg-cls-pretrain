from __future__ import annotations

from typing import Dict, List

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .splits import list_pairs


def build_seg_tf(image_size: int, train: bool):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
                A.Affine(
                    translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                    scale=(0.9, 1.1),  # 1.0 is identity, so 0.9 to 1.1 is +/- 10%
                    rotate=(-45, 45),
                    p=0.5,
                ),
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )


def build_cls_tf(image_size: int, train: bool):
    # 分类只处理 image，不需要 mask
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
                A.RandomResizedCrop(
                    (image_size, image_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.6
                ),
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )


def build_pretrain_tf(image_size: int):
    # SimCLR 风格：增强更强一些
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
    # PIL(RGB) -> np.uint8(H,W,3) RGB
    return np.array(img, dtype=np.uint8)


def pil_mask_to_np(mask: Image.Image) -> np.ndarray:
    # PIL(P) or L -> np.uint8(H,W)
    return np.array(mask, dtype=np.uint8)


class SegFromList(Dataset):
    """
    items: {"image": "...", "mask": "...", "label": 0/1}
    mask: 1=物体,2=背景,3=边缘
    输出二分类 mask01：前景=(1或3)
    """

    def __init__(self, items: List[Dict], tf):
        self.items = items
        self.tf = tf  # albumentations.Compose

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        mask = Image.open(it["mask"])

        img_np = pil_rgb_to_np(img)
        mask_np = pil_mask_to_np(mask)

        # 二分类：把 1/3 都当作前景
        mask01 = ((mask_np == 1) | (mask_np == 3)).astype(np.uint8)  # {0,1}

        aug = self.tf(image=img_np, mask=mask01)
        x = aug["image"]  # torch.FloatTensor [3,H,W]
        m = aug["mask"].long()  # torch.LongTensor [H,W]
        return x, m


class ClsFromList(Dataset):
    def __init__(self, items: List[Dict], tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        img_np = pil_rgb_to_np(img)
        aug = self.tf(image=img_np)
        x = aug["image"]
        y = int(it["label"])
        return x, y


class PretrainImages(Dataset):
    """
    返回两份增强视图 (x1,x2)
    """

    def __init__(self, image_paths, tf):
        self.paths = image_paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        img_np = pil_rgb_to_np(img)
        x1 = self.tf(image=img_np)["image"]
        x2 = self.tf(image=img_np)["image"]
        return x1, x2


def build_seg_loaders(
    split_dict: dict, image_size: int, batch_size: int, num_workers: int
):
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
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )


def build_cls_loaders(
    split_dict: dict, image_size: int, batch_size: int, num_workers: int
):
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
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )


def build_pretrain_loader(
    data_root: str, image_size: int, batch_size: int, num_workers: int
):
    tf = build_pretrain_tf(image_size)
    imgs, _ = list_pairs(data_root, "pretrain")
    ds = PretrainImages(imgs, tf)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
