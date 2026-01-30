from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def list_pairs(data_root: str, subset: str) -> Tuple[List[Path], List[Path] | None]:
    """
    subset: "cat"/"dog"/"pretrain"
    Returns image paths and mask paths (pretrain has no masks).
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
    mask_map = {p.stem: p for p in masks}
    pairs = []
    for ip in imgs:
        mp = mask_map.get(ip.stem)
        if mp is None:
            continue
        pairs.append((ip, mp))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def make_splits_for_catdog(
    data_root: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    out_json: str,
) -> str:
    """Create train/val/test split for cat+dog paired data."""
    cat_imgs, cat_masks = list_pairs(data_root, "cat")
    dog_imgs, dog_masks = list_pairs(data_root, "dog")

    imgs = cat_imgs + dog_imgs
    masks = cat_masks + dog_masks
    labels = [0] * len(cat_imgs) + [1] * len(dog_imgs)

    n = len(imgs)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    test_n = int(round(n * test_ratio))
    val_n = int(round(n * val_ratio))
    test_idx = perm[:test_n]
    val_idx = perm[test_n : test_n + val_n]
    train_idx = perm[test_n + val_n :]

    def pack(idxs: List[int]) -> List[Dict[str, str | int]]:
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

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def load_splits(json_path: str) -> Dict:
    path = Path(json_path)
    return json.loads(path.read_text(encoding="utf-8"))
