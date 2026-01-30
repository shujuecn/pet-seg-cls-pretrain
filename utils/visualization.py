"""Visualization helpers for segmentation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

PALETTE = [
    (0, 0, 0),        # class 0
    (255, 128, 0),    # class 1
    (0, 200, 255),    # class 2
]


def mask_to_color(mask: np.ndarray, palette: List[Tuple[int, int, int]] | None = None) -> Image.Image:
    palette = palette or PALETTE
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in enumerate(palette):
        color[mask == idx] = rgb
    return Image.fromarray(color)


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    color = np.array(mask_to_color(mask))
    base = np.array(image.convert("RGB"))
    out = (base * (1 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(out)


def save_index_md(items: Iterable[Tuple[str, str, str, str]], out_path: Path) -> None:
    lines = ["# Segmentation Results", "", "|Image|GT|Pred|Overlay|", "|---|---|---|---|"]
    for img, gt, pred, overlay in items:
        lines.append(
            f"|![]({img})|![]({gt})|![]({pred})|![]({overlay})|"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
