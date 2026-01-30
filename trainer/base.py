from __future__ import annotations

from typing import Any

import torch


class BaseTrainer:
    def __init__(self, device: str):
        self.device = device

    def to_device(self, batch: Any):
        if isinstance(batch, (list, tuple)):
            return [self.to_device(x) for x in batch]
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        return batch
