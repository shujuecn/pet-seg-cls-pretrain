"""Configuration management with Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, validator


class SplitConfig(BaseModel):
    val_ratio: float = Field(0.15, ge=0.0, le=0.5)
    test_ratio: float = Field(0.15, ge=0.0, le=0.5)


class TaskConfig(BaseModel):
    run_seg: bool = False
    run_cls: bool = False
    run_pretrain: bool = False
    run_pretrained_encoder_to_seg: bool = True


class OptimConfig(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = Field(0.0, ge=0.0)
    grad_accum_steps: int = Field(1, ge=1)


class SchedulerConfig(BaseModel):
    type: Literal["none", "cosine", "plateau"] = "none"
    t_max: int = 10
    eta_min: float = 1e-6
    plateau_factor: float = 0.5
    plateau_patience: int = 3


class EpochsConfig(BaseModel):
    seg: int = 12
    cls: int = 8
    pretrain: int = 10
    seg_ft: int = 10
    freeze_encoder_epochs: int = 2


class ModelConfig(BaseModel):
    base_channels: int = 32
    depth: int = 4
    num_classes: int = 3


class PretrainConfig(BaseModel):
    proj_dim: int = 128
    temperature: float = 0.2


class LossConfig(BaseModel):
    seg_ce_weight: float = 1.0
    seg_dice_weight: float = 0.5


class DataLoaderConfig(BaseModel):
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


class LoggingConfig(BaseModel):
    log_images_every: int = 5
    num_visual_samples: int = 4


class SegOutputConfig(BaseModel):
    save_original_size: bool = True
    pred_mask_values: Literal["zero_based", "one_based"] = "zero_based"
    save_overlay: bool = True


class ClsOutputConfig(BaseModel):
    save_topk_errors: bool = True
    topk_errors: int = 12


class AppConfig(BaseModel):
    data_root: str = "./data"
    image_size: int = 256
    batch_size: int = 16
    pretrain_batch_size: int = 128
    num_workers: int = 8
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    exp_name: str = "default"
    output_dir: str = "./runs"
    rebuild_split: bool = False

    split: SplitConfig = SplitConfig()
    task: TaskConfig = TaskConfig()
    optim: OptimConfig = OptimConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    epochs: EpochsConfig = EpochsConfig()
    model: ModelConfig = ModelConfig()
    pretrain: PretrainConfig = PretrainConfig()
    loss: LossConfig = LossConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()
    logging: LoggingConfig = LoggingConfig()
    seg_output: SegOutputConfig = SegOutputConfig()
    cls_output: ClsOutputConfig = ClsOutputConfig()

    @validator("device")
    def _normalize_device(cls, v: str) -> str:
        return v.lower()

    @validator("output_dir")
    def _normalize_output_dir(cls, v: str) -> str:
        return str(Path(v))

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.exp_name


def load_config(path: str) -> AppConfig:
    """Load configuration from YAML and validate."""
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return AppConfig(**payload)


def save_config(cfg: AppConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.dict(), f, sort_keys=False, allow_unicode=True)
