from __future__ import annotations

import gc
import os
import platform
import shutil
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import (
    build_cls_loaders,
    build_pretrain_loader,
    build_seg_loaders,
    load_splits,
    make_splits_for_catdog,
)
from models import EncoderClassifier, EncoderSimCLR, UNet, load_encoder_weights
from trainer import ClsTrainer, PretrainTrainer, SegTrainer
from utils import load_config, save_config, seed_everything, setup_logger

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze_module(module: torch.nn.Module, freeze: bool) -> None:
    for p in module.parameters():
        p.requires_grad = not freeze


def main() -> None:
    cfg = load_config("./config.yaml")

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    seed_everything(cfg.seed)

    base_run_dir = cfg.run_dir()
    base_run_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = platform.node() or "host"
    run_dir = base_run_dir / f"{host}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    save_config(cfg, run_dir / "config.yaml")
    logger = setup_logger(run_dir / "train.log")
    logger.info("Running on device: %s", device)
    logger.info("Config: %s", cfg.model_dump_json(indent=2))

    base_split_json = base_run_dir / "split.json"
    if cfg.rebuild_split or not base_split_json.exists():
        make_splits_for_catdog(
            data_root=cfg.data_root,
            val_ratio=cfg.split.val_ratio,
            test_ratio=cfg.split.test_ratio,
            seed=cfg.seed,
            out_json=str(base_split_json),
        )

    # 每次运行把 split 复制到本次 run_dir
    split_json = run_dir / "split.json"
    shutil.copy2(base_split_json, split_json)

    split_dict = load_splits(str(split_json))
    logger.info("Loaded splits: %s", split_dict.get("meta"))

    seg_train, seg_val, seg_test = build_seg_loaders(
        split_dict,
        cfg.image_size,
        cfg.batch_size,
        cfg.num_workers,
        cfg.dataloader.pin_memory,
        cfg.dataloader.prefetch_factor,
        cfg.dataloader.persistent_workers,
        cfg.seed,
    )
    cls_train, cls_val, cls_test = build_cls_loaders(
        split_dict,
        cfg.image_size,
        cfg.batch_size,
        cfg.num_workers,
        cfg.dataloader.pin_memory,
        cfg.dataloader.prefetch_factor,
        cfg.dataloader.persistent_workers,
        cfg.seed,
    )
    pre_loader = build_pretrain_loader(
        cfg.data_root,
        cfg.image_size,
        cfg.pretrain_batch_size,
        cfg.num_workers,
        cfg.dataloader.pin_memory,
        cfg.dataloader.prefetch_factor,
        cfg.dataloader.persistent_workers,
        cfg.seed,
    )

    base_ch = cfg.model.base_channels
    depth = cfg.model.depth

    writer = SummaryWriter(log_dir=run_dir.as_posix())

    if cfg.task.run_seg:
        unet = UNet(
            num_classes=cfg.model.num_classes, base_channels=base_ch, depth=depth
        )
        seg_trainer = SegTrainer(
            unet,
            cfg.optim,
            cfg.loss,
            cfg.scheduler,
            device,
            cfg.amp,
            cfg.model.num_classes,
            run_dir,
            writer,
            logger,
        )
        ckpt_path = str(run_dir / "unet_scratch.pt")
        seg_trainer.fit(
            seg_train,
            seg_val,
            cfg.epochs.seg,
            ckpt_path,
            cfg.logging.log_images_every,
            cfg.logging.num_visual_samples,
        )
        metrics = seg_trainer.test_and_save(
            seg_test,
            run_dir / "seg_test",
            cfg.seg_output.save_original_size,
            cfg.seg_output.pred_mask_values,
            cfg.seg_output.save_overlay,
        )
        logger.info("[seg] test metrics: %s", metrics)
        del seg_trainer, unet
        clear_cuda()

    if cfg.task.run_cls:
        clf = EncoderClassifier(num_classes=2, base_channels=base_ch, depth=depth)
        cls_trainer = ClsTrainer(
            clf,
            cfg.optim,
            cfg.scheduler,
            device,
            cfg.amp,
            run_dir,
            writer,
            logger,
        )
        ckpt_path = str(run_dir / "encoder_cls.pt")
        cls_trainer.fit(cls_train, cls_val, cfg.epochs.cls, ckpt_path)
        metrics = cls_trainer.test_and_save(
            cls_test,
            run_dir,
            cfg.cls_output.save_topk_errors,
            cfg.cls_output.topk_errors,
        )
        logger.info("[cls] test metrics: %s", metrics)
        del cls_trainer, clf
        clear_cuda()

    if cfg.task.run_pretrain:
        simclr = EncoderSimCLR(
            base_channels=base_ch, depth=depth, proj_dim=cfg.pretrain.proj_dim
        )
        pre_trainer = PretrainTrainer(
            simclr,
            cfg.optim,
            cfg.scheduler,
            device,
            temperature=cfg.pretrain.temperature,
            amp=cfg.amp,
            run_dir=run_dir,
            writer=writer,
            logger=logger,
        )
        ckpt_path = str(run_dir / "encoder_simclr.pt")
        pre_trainer.fit(pre_loader, cfg.epochs.pretrain, ckpt_path)
        del pre_trainer, simclr
        clear_cuda()

    if cfg.task.run_pretrained_encoder_to_seg:
        unet = UNet(
            num_classes=cfg.model.num_classes, base_channels=base_ch, depth=depth
        )
        encoder_src = None
        simclr_path = run_dir / "encoder_simclr.pt"
        cls_path = run_dir / "encoder_cls.pt"
        if simclr_path.exists():
            encoder_src = simclr_path
        elif cls_path.exists():
            encoder_src = cls_path

        if encoder_src is None:
            logger.warning("No pretrained encoder checkpoint found, skipping.")
        else:
            load_encoder_weights(unet.encoder, str(encoder_src), strict=False)

            seg_trainer = SegTrainer(
                unet,
                cfg.optim,
                cfg.loss,
                cfg.scheduler,
                device,
                cfg.amp,
                cfg.model.num_classes,
                run_dir,
                writer,
                logger,
            )

            freeze_ep = cfg.epochs.freeze_encoder_epochs
            seg_ft_ep = cfg.epochs.seg_ft

            if freeze_ep > 0:
                freeze_module(seg_trainer.model.encoder, True)
                seg_trainer.fit(
                    seg_train,
                    seg_val,
                    freeze_ep,
                    str(run_dir / "unet_pretrained_frozen.pt"),
                    cfg.logging.log_images_every,
                    cfg.logging.num_visual_samples,
                )

            freeze_module(seg_trainer.model.encoder, False)
            for g in seg_trainer.opt.param_groups:
                g["lr"] *= 0.5

            seg_trainer.fit(
                seg_train,
                seg_val,
                seg_ft_ep,
                str(run_dir / "unet_pretrained_ft.pt"),
                cfg.logging.log_images_every,
                cfg.logging.num_visual_samples,
            )
            metrics = seg_trainer.test_and_save(
                seg_test,
                run_dir / "seg_test_pretrained",
                cfg.seg_output.save_original_size,
                cfg.seg_output.pred_mask_values,
                cfg.seg_output.save_overlay,
            )
            logger.info("[seg_pretrained] test metrics: %s", metrics)
        del unet
        clear_cuda()

    writer.close()


if __name__ == "__main__":
    main()
