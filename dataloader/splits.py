from __future__ import annotations
import os
from pathlib import Path
import json
import torch


def list_pairs(data_root: str, subset: str):
    """
    subset: 'cat'/'dog'/'pretrain'
    返回 image_paths, mask_paths（pretrain 只有 image）
    """
    root = Path(data_root)
    img_dir = root / "image" / subset
    mask_dir = root / "mask" / subset

    imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if subset == "pretrain":
        return imgs, None

    masks = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"])
    # 以 stem 对齐
    mask_map = {p.stem: p for p in masks}
    pairs = []
    for ip in imgs:
        mp = mask_map.get(ip.stem)
        if mp is None:
            continue
        pairs.append((ip, mp))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def make_splits_for_catdog(
    data_root: str, val_ratio: float, test_ratio: float, seed: int, out_json: str
):
    """
    对 cat+dog 的 paired 数据做 Train/Val/Test
    保存为 json，后续复现实验
    """
    cat_imgs, cat_masks = list_pairs(data_root, "cat")
    dog_imgs, dog_masks = list_pairs(data_root, "dog")

    imgs = cat_imgs + dog_imgs
    masks = cat_masks + dog_masks
    labels = [0] * len(cat_imgs) + [1] * len(dog_imgs)  # 分类标签：cat=0,dog=1

    n = len(imgs)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    test_n = int(round(n * test_ratio))
    val_n = int(round(n * val_ratio))
    test_idx = perm[:test_n]
    val_idx = perm[test_n : test_n + val_n]
    train_idx = perm[test_n + val_n :]

    def pack(idxs):
        return [
            {
                "image": str(imgs[i]),
                "mask": str(masks[i]),
                "label": labels[i],
            }
            for i in idxs
        ]

    payload = {
        "train": pack(train_idx),
        "val": pack(val_idx),
        "test": pack(test_idx),
        "meta": {
            "n": n,
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_json


def load_splits(json_path: str):
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
