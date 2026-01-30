import gc
import os

import torch
import yaml

from dataloader.pet_ds import (
    build_cls_loaders,
    build_pretrain_loader,
    build_seg_loaders,
)
from dataloader.splits import load_splits, make_splits_for_catdog
from models.heads import EncoderClassifier, EncoderSimCLR
from models.unet import UNet
from trainer.cls_trainer import ClsTrainer
from trainer.pretrain_trainer import PretrainTrainer
from trainer.seg_trainer import SegTrainer
from trainer.utils import set_seed


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze_module(m, freeze: bool):
    for p in m.parameters():
        p.requires_grad = not freeze


def main():
    cfg = yaml.safe_load(open("./config.yaml", "r", encoding="utf-8"))

    set_seed(cfg["seed"])
    device = (
        cfg["device"]
        if torch.cuda.is_available() and cfg["device"] == "cuda"
        else "cpu"
    )
    data_root = cfg["data_root"]
    os.makedirs("./runs", exist_ok=True)

    # 1) 生成/加载 split（只对 cat+dog 的 1600 paired 做划分）
    split_json = "./runs/split_catdog.json"
    if not os.path.exists(split_json):
        make_splits_for_catdog(
            data_root=data_root,
            val_ratio=cfg["split"]["val_ratio"],
            test_ratio=cfg["split"]["test_ratio"],
            seed=cfg["seed"],
            out_json=split_json,
        )
    split_dict = load_splits(split_json)
    print("Loaded splits:", split_dict["meta"])

    # 2) loaders
    seg_train, seg_val, seg_test = build_seg_loaders(
        split_dict, cfg["image_size"], cfg["batch_size"], cfg["num_workers"]
    )
    cls_train, cls_val, cls_test = build_cls_loaders(
        split_dict, cfg["image_size"], cfg["batch_size"], cfg["num_workers"]
    )
    pre_loader = build_pretrain_loader(
        data_root, cfg["image_size"], cfg["pretrain_batch_size"], cfg["num_workers"]
    )

    base_ch = cfg["model"]["base_channels"]
    depth = cfg["model"]["depth"]

    # (A) 分割从头
    if cfg["task"]["run_seg"]:
        unet = UNet(num_classes=1, base_channels=base_ch, depth=depth)
        seg_trainer = SegTrainer(
            unet, cfg["optim"]["lr"], cfg["optim"]["weight_decay"], device
        )
        seg_trainer.fit(
            seg_train, seg_val, cfg["epochs"]["seg"], ckpt_path="./runs/unet_scratch.pt"
        )
        test = seg_trainer.evaluate(seg_test)
        print("[seg] test:", test)

        del seg_trainer, unet
        clear_cuda()

    # (B) Encoder 分类（监督预训练的一种）
    if cfg["task"]["run_cls"]:
        clf = EncoderClassifier(num_classes=2, base_channels=base_ch, depth=depth)
        cls_trainer = ClsTrainer(
            clf, cfg["optim"]["lr"], cfg["optim"]["weight_decay"], device
        )
        cls_trainer.fit(
            cls_train, cls_val, cfg["epochs"]["cls"], ckpt_path="./runs/encoder_cls.pt"
        )
        test = cls_trainer.evaluate(cls_test)
        print("[cls] test:", test)

        del cls_trainer, clf
        clear_cuda()

    # (C) Encoder 自监督预训练（用 pretrain 图片）
    if cfg["task"]["run_pretrain"]:
        simclr = EncoderSimCLR(
            base_channels=base_ch, depth=depth, proj_dim=cfg["pretrain"]["proj_dim"]
        )
        pre_trainer = PretrainTrainer(
            simclr,
            cfg["optim"]["lr"],
            cfg["optim"]["weight_decay"],
            device,
            temperature=cfg["pretrain"]["temperature"],
        )
        pre_trainer.fit(
            pre_loader, cfg["epochs"]["pretrain"], ckpt_path="./runs/encoder_simclr.pt"
        )

        del pre_trainer, simclr
        clear_cuda()

    # (D) 用预训练 encoder 初始化 U-Net 再做分割（迁移学习演示）
    if cfg["task"]["run_pretrained_encoder_to_seg"]:
        unet = UNet(num_classes=1, base_channels=base_ch, depth=depth)

        # 这里选择用哪种预训练：优先 SimCLR，否则用分类 encoder
        encoder_src = (
            "./runs/encoder_simclr.pt"
            if os.path.exists("./runs/encoder_simclr.pt")
            else "./runs/encoder_cls.pt"
        )
        ckpt = torch.load(encoder_src, map_location="cpu", weights_only=True)["model"]

        # 根据保存的模型结构取 encoder 权重
        # SimCLR: keys like "encoder.stem...." / Cls: "encoder.stem...."
        enc_state = {
            k.replace("encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }
        unet.encoder.load_state_dict(enc_state, strict=True)

        # 冻结-解冻策略
        freeze_ep = cfg["epochs"]["freeze_encoder_epochs"]
        seg_ft_ep = cfg["epochs"]["seg_ft"]

        seg_trainer = SegTrainer(
            unet, cfg["optim"]["lr"], cfg["optim"]["weight_decay"], device
        )

        if freeze_ep > 0:
            freeze_module(seg_trainer.model.encoder, True)
            seg_trainer.fit(
                seg_train,
                seg_val,
                freeze_ep,
                ckpt_path="./runs/unet_pretrained_frozen.pt",
            )

        freeze_module(seg_trainer.model.encoder, False)
        # fine-tune 学习率可更小，这里简单减半
        for g in seg_trainer.opt.param_groups:
            g["lr"] *= 0.5

        seg_trainer.fit(
            seg_train, seg_val, seg_ft_ep, ckpt_path="./runs/unet_pretrained_ft.pt"
        )
        test = seg_trainer.evaluate(seg_test)
        print("[seg_pretrained] test:", test)


if __name__ == "__main__":
    main()
