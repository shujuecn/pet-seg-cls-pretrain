from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Model definition (same as your unet.py)
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = ConvBNReLU(in_ch, out_ch)
        self.c2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.c2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.c1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.c2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, base_channels: int = 32, depth: int = 4, in_channels: int = 3):
        super().__init__()
        self.depth = depth
        chs = [base_channels * (2**i) for i in range(depth)]
        self.stem = DownBlock(in_channels, chs[0])
        self.downs = nn.ModuleList([DownBlock(chs[i - 1], chs[i]) for i in range(1, depth)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DownBlock(chs[-1], chs[-1] * 2)
        self.out_channels = chs[-1] * 2
        self.skip_channels = chs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        x = self.stem(x)
        skips.append(x)
        for blk in self.downs:
            x = self.pool(x)
            x = blk(x)
            skips.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        return x, skips


class UNet(nn.Module):
    def __init__(self, num_classes: int = 3, base_channels: int = 32, depth: int = 4, in_channels: int = 3):
        super().__init__()
        self.encoder = UNetEncoder(base_channels, depth, in_channels)
        skip_chs = self.encoder.skip_channels
        cur = self.encoder.out_channels
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(UpBlock(cur, skip_chs[i], skip_chs[i]))
            cur = skip_chs[i]
        self.head = nn.Conv2d(cur, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        for up, i in zip(self.ups, reversed(range(len(skips)))):
            x = up(x, skips[i])
        return self.head(x)


# -------------------------
# Utilities
# -------------------------
def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def resize_keep_aspect(img: Image.Image, target: int) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """Resize so that longer side == target, keep aspect."""
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img2 = img.resize((nw, nh), resample=Image.BILINEAR)
    return img2, scale, (w, h)


def pad_to_multiple(img: Image.Image, multiple: int = 2**4) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Pad right/bottom to make H,W multiples of `multiple`.
    Return padded img and pad (left, top, right, bottom).
    """
    w, h = img.size
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple
    if pad_w == 0 and pad_h == 0:
        return img, (0, 0, 0, 0)
    new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    new_img.paste(img, (0, 0))
    return new_img, (0, 0, pad_w, pad_h)


def unpad_np(arr: np.ndarray, pad: Tuple[int, int, int, int]) -> np.ndarray:
    _, _, pr, pb = pad
    if pr == 0 and pb == 0:
        return arr
    h, w = arr.shape[:2]
    return arr[: h - pb, : w - pr]


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    # simple normalization (ImageNet-ish). You can replace with your train-time norm if different.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)

def to_tensor_fix(img: Image.Image) -> torch.Tensor:
    # TODO: use the same normalization as training
    arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: HxW, values {0,1,2}
    return: HxWx3 uint8
    """
    palette = {
        0: (0, 0, 0),        # background
        1: (255, 80, 80),    # object
        2: (80, 160, 255),   # border/edge
    }
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in palette.items():
        out[mask == k] = np.array(v, dtype=np.uint8)
    return out


def overlay_on_image(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    m = mask_rgb.astype(np.float32)
    out = img * (1 - alpha) + m * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# Feature hook & saving
# -------------------------
def resolve_modules(model: nn.Module, names: List[str]) -> List[Tuple[str, nn.Module]]:
    table = dict(model.named_modules())
    out = []
    for n in names:
        if n not in table:
            raise KeyError(
                f"Layer '{n}' not found. Available examples: "
                f"{list(sorted(table.keys()))[:30]} ... (use --list_layers)"
            )
        out.append((n, table[n]))
    return out


class FeatureCatcher:
    def __init__(self):
        self.feats: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def add(self, name: str, module: nn.Module):
        def hook_fn(_m, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.feats[name] = out.detach().cpu()
        h = module.register_forward_hook(hook_fn)
        self.hooks.append(h)

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def save_feature_grid(
    feat: torch.Tensor,
    out_path: Path,
    max_channels: int = 32,
    cols: int = 8,
) -> None:
    """
    feat: (N,C,H,W) or (C,H,W)
    Save a grid image of first N=0 sample channels.
    """
    if feat.ndim == 4:
        feat = feat[0]
    if feat.ndim != 3:
        return
    c, h, w = feat.shape
    k = min(c, max_channels)

    x = feat[:k]  # (k,H,W)
    # per-channel normalize to 0..255
    x = x.numpy()
    imgs = []
    for i in range(k):
        a = x[i]
        a = a - a.min()
        if a.max() > 1e-6:
            a = a / a.max()
        a = (a * 255.0).astype(np.uint8)
        imgs.append(a)

    rows = int(math.ceil(k / cols))
    grid = Image.new("L", (cols * w, rows * h), 0)
    for idx, im in enumerate(imgs):
        r, c0 = divmod(idx, cols)
        grid.paste(Image.fromarray(im, mode="L"), (c0 * w, r * h))

    ensure_dir(out_path.parent)
    grid.save(out_path)


# -------------------------
# Checkpoint loading
# -------------------------
def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            # assume it's already a state_dict-like mapping
            state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    # allow both "model.xxx" and raw keys
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:20]}{' ...' if len(missing)>20 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:20]}{' ...' if len(unexpected)>20 else ''}")


# -------------------------
# Main inference
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./runs/demo/WIN-20240402CTJ-20260130-224248/unet_scratch.pt", help="Path to UNet checkpoint (.pt)")
    ap.add_argument("--image", type=str, default="./outputs/Abyssinian_136.png", help="Path to input image")
    ap.add_argument("--out_dir", type=str, default="./outputs/unet_test", help="Output directory")
    ap.add_argument("--image_size", type=int, default=256, help="Resize long side to this before padding")
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true", help="Use AMP for inference (cuda only)")

    ap.add_argument(
        "--layers",
        type=str,
        default="encoder.stem,encoder.downs.0,encoder.downs.1,encoder.bottleneck,ups.3,ups.2,ups.1,ups.0,head",
        help="Comma separated layer names to dump feature maps",
    )
    ap.add_argument("--max_channels", type=int, default=32, help="Max channels per layer to save")
    ap.add_argument("--cols", type=int, default=8, help="Columns in feature grid")
    ap.add_argument("--list_layers", action="store_true", help="Print all named_modules and exit")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    img_path = Path(args.image)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    model = UNet(
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        depth=args.depth,
        in_channels=3,
    )
    model.eval()
    load_checkpoint(model, ckpt_path)
    model.to(device)

    if args.list_layers:
        for n, _m in model.named_modules():
            print(n)
        return

    # prepare image
    img0 = load_image_rgb(img_path)
    img_rs, scale, orig_wh = resize_keep_aspect(img0, args.image_size)
    img_pad, pad = pad_to_multiple(img_rs, multiple=2 ** args.depth)

    x = to_tensor(img_pad).unsqueeze(0).to(device)

    # feature hooks
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    catcher = FeatureCatcher()
    for name, module in resolve_modules(model, layers):
        catcher.add(name, module)

    # forward
    with torch.inference_mode():
        if args.amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)

    catcher.close()

    # prediction mask in padded-resized space
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # (H,W)

    # unpad back to resized
    pred = unpad_np(pred, pad)

    # resize back to original size
    pred_img = Image.fromarray(pred, mode="L")
    pred_img = pred_img.resize(orig_wh, resample=Image.NEAREST)
    pred_np = np.asarray(pred_img).astype(np.uint8)

    # save mask
    pred_mask_path = out_dir / "pred_mask.png"
    Image.fromarray(pred_np, mode="L").save(pred_mask_path)

    # overlay
    mask_rgb = colorize_mask(pred_np)
    img_np = np.asarray(img0).astype(np.uint8)
    overlay = overlay_on_image(img_np, mask_rgb, alpha=0.45)
    Image.fromarray(overlay).save(out_dir / "overlay.png")

    # save color mask too
    Image.fromarray(mask_rgb).save(out_dir / "pred_mask_color.png")

    # feature maps
    fmap_root = out_dir / "featuremaps"
    for lname, feat in catcher.feats.items():
        # save one grid image per layer
        out_path = fmap_root / lname.replace(".", "_") / "grid.png"
        save_feature_grid(feat, out_path, max_channels=args.max_channels, cols=args.cols)

    print(f"[ok] saved: {pred_mask_path}")
    print(f"[ok] saved: {out_dir / 'overlay.png'}")
    print(f"[ok] featuremaps: {fmap_root}")


if __name__ == "__main__":
    main()
