"""Model helpers for loading checkpoints."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _strip_prefix(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def extract_encoder_state(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract encoder weights from different checkpoint formats."""
    if any(k.startswith("encoder.") for k in state_dict):
        return _strip_prefix(state_dict, "encoder.")
    if any(k.startswith("module.encoder.") for k in state_dict):
        return _strip_prefix(state_dict, "module.encoder.")
    if any(k.startswith("model.encoder.") for k in state_dict):
        return _strip_prefix(state_dict, "model.encoder.")
    return state_dict


def load_encoder_weights(
    encoder: nn.Module,
    ckpt_path: str,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load encoder weights from a SimCLR/Cls checkpoint into UNet encoder."""
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    enc_state = extract_encoder_state(state)
    encoder.load_state_dict(enc_state, strict=strict)
    return enc_state
