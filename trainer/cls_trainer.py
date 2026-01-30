import torch
import torch.nn as nn
from tqdm import tqdm
from .base import BaseTrainer
from .utils import save_ckpt


class ClsTrainer(BaseTrainer):
    def __init__(self, model, lr, weight_decay, device: str, amp: bool = True):
        super().__init__(device)
        self.model = model.to(device)
        self.crit = nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )

        self.amp = bool(amp) and (device == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        losses = []
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="[cls] eval", leave=False)
        for x, y in pbar:
            x, y = self.to_device([x, y])
            logits = self.model(x)
            loss = self.crit(logits, y)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            losses.append(loss.item())
            pbar.set_postfix(loss=float(loss.item()))
        return {
            "loss": sum(losses) / max(1, len(losses)),
            "acc": correct / max(1, total),
        }

    def fit(self, train_loader, val_loader, epochs: int, ckpt_path: str):
        best = -1.0
        for ep in range(1, epochs + 1):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"[cls] {ep}/{epochs}", leave=False)
            for x, y in pbar:
                x, y = self.to_device([x, y])

                self.opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                    logits = self.model(x)
                    loss = self.crit(logits, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                pbar.set_postfix(loss=float(loss.item()))

            val = self.evaluate(val_loader)
            print(f"[cls] ep{ep}: val_loss={val['loss']:.4f} acc={val['acc']:.4f}")
            if val["acc"] > best:
                best = val["acc"]
                save_ckpt(ckpt_path, self.model, extra={"best_acc": best})
        print(f"[cls] best_acc={best:.4f} saved: {ckpt_path}")
