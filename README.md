# pet-seg-cls-pretrain

一个教学友好的 PyTorch 项目，覆盖三类语义分割、猫狗分类、以及 SimCLR 自监督预训练。项目重点：结构清晰、可复现、可扩展、训练稳定、日志与实验管理规范。

## 1. 环境安装

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果你没有 requirements.txt，可以安装以下依赖（按需调整版本）：

```bash
pip install torch torchvision albumentations opencv-python pydantic pyyaml tensorboard
```

## 2. 数据准备

数据目录结构：

```
./data/
  image/
    cat/        # 800
    dog/        # 800
    pretrain/   # 自监督图片
  mask/
    cat/        # png mask，对应 image/cat
    dog/        # png mask，对应 image/dog
```

- 分割 mask: 1=物体，2=背景，3=边缘。
- 训练时会统一映射为 {0,1,2}（详见 dataloader/pet_ds.py）。

## 3. 运行方式

配置文件统一写在 `config.yaml`，不使用 argparse。你可以修改：

- `exp_name`：实验名（输出到 `runs/exp_name/`）
- `rebuild_split`：是否重建 split
- `task.*`：开启需要的任务

### 3.1 三类语义分割（从头训练）

```yaml
# config.yaml
 task:
   run_seg: true
   run_cls: false
   run_pretrain: false
   run_pretrained_encoder_to_seg: false
```

```bash
python train.py
```

输出：
- `runs/exp_name/unet_scratch.pt` (best checkpoint)
- `runs/exp_name/seg_test/` (test 可视化、mask、summary.md)

### 3.2 猫狗分类

```yaml
 task:
   run_seg: false
   run_cls: true
   run_pretrain: false
```

```bash
python train.py
```

输出：
- `runs/exp_name/encoder_cls.pt`
- `runs/exp_name/cls_test.csv`
- `runs/exp_name/cls_errors/` (可选错分样例)

### 3.3 SimCLR 自监督预训练

```yaml
 task:
   run_pretrain: true
```

```bash
python train.py
```

输出：
- `runs/exp_name/encoder_simclr.pt`
- TensorBoard loss 曲线

### 3.4 预训练 encoder → 分割微调

```yaml
 task:
   run_pretrained_encoder_to_seg: true
```

- 会优先使用 `runs/exp_name/encoder_simclr.pt`，否则使用 `encoder_cls.pt`。
- 支持冻结 encoder 训练若干 epoch，再解冻微调。

## 4. 复现与日志

- split 文件：`runs/exp_name/split.json`（默认复用，可 `rebuild_split: true` 重新生成）
- 训练日志：`runs/exp_name/train.log`
- TensorBoard：`runs/exp_name/tensorboard/`

启动 TensorBoard：

```bash
tensorboard --logdir runs
```

## 5. 关键配置说明

- `optim.grad_clip_norm`：梯度裁剪
- `optim.grad_accum_steps`：梯度累积
- `scheduler.type`：`none/cosine/plateau`
- `loss.seg_ce_weight` + `loss.seg_dice_weight`：多类分割损失组合
- `seg_output.pred_mask_values`：保存 mask 为 `0/1/2` 或 `1/2/3`

## 6. 常见问题

### (1) Windows: safe.directory 报错

Git 可能提示：

```
fatal: detected dubious ownership in repository
```

解决方法：

```bash
git config --global --add safe.directory "<your_repo_path>"
```

### (2) AMP 半精度溢出

如果出现 loss 变成 `inf`/`nan`，建议：
- 降低 `optim.lr`
- 增大 `grad_accum_steps`
- 关闭 AMP：`amp: false`

### (3) Albumentations 联网 warning

Albumentations 会提示版本检查，需要联网。可以忽略，或设置：

```bash
export NO_ALBUMENTATIONS_UPDATE=1
```

## 7. 目录结构

```
.
├── dataloader/  # 数据加载与增强
├── models/      # U-Net / 分类 / SimCLR
├── trainer/     # Trainer 与评估
├── utils/       # 配置、日志、可视化、复现
├── train.py     # 统一训练入口
├── config.yaml  # 配置文件
└── runs/        # 输出目录
```
