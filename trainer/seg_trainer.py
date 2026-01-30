import torch
from tqdm import tqdm
from .base import BaseTrainer
from .utils import DiceBCELoss
from .utils import dice_coeff, iou_score
from .utils import save_ckpt


class SegTrainer(BaseTrainer):
    def __init__(self, model, lr, weight_decay, device: str, amp: bool = True):
        super().__init__(device)
        self.model = model.to(device)
        self.crit = DiceBCELoss()
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )

        self.amp = bool(amp) and (device == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        losses, dices, ious = [], [], []
        for x, m in loader:
            x, m = self.to_device([x, m])
            with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                logits = self.model(x)
                loss = self.crit(logits, m)
            pred = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
            losses.append(loss.item())
            dices.append(dice_coeff(pred, m))
            ious.append(iou_score(pred, m))
        return {
            "loss": sum(losses) / max(1, len(losses)),
            "dice": sum(dices) / max(1, len(dices)),
            "iou": sum(ious) / max(1, len(ious)),
        }

    def fit(self, train_loader, val_loader, epochs: int, ckpt_path: str):
        best = -1.0
        for ep in range(1, epochs + 1):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"[seg] {ep}/{epochs}", leave=False)
            for x, m in pbar:
                x, m = self.to_device([x, m])

                self.opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                    logits = self.model(x)
                    loss = self.crit(logits, m)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                pbar.set_postfix(loss=float(loss.item()))

            val = self.evaluate(val_loader)
            print(
                f"[seg] ep{ep}: val_loss={val['loss']:.4f} dice={val['dice']:.4f} iou={val['iou']:.4f}"
            )
            if val["dice"] > best:
                best = val["dice"]
                save_ckpt(ckpt_path, self.model, extra={"best_dice": best})
        print(f"[seg] best_dice={best:.4f} saved: {ckpt_path}")
