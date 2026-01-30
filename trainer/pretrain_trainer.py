import torch
from tqdm import tqdm
from .base import BaseTrainer
from .utils import nt_xent_loss
from .utils import save_ckpt


class PretrainTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        lr,
        weight_decay,
        device: str,
        temperature: float = 0.2,
        amp: bool = True,
    ):
        super().__init__(device)
        self.model = model.to(device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        self.temperature = temperature

        self.amp = bool(amp) and (device == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

    def fit(self, loader, epochs: int, ckpt_path: str):
        best = 1e9
        for ep in range(1, epochs + 1):
            self.model.train()
            losses = []
            pbar = tqdm(loader, desc=f"[pretrain] {ep}/{epochs}", leave=False)
            for x1, x2 in pbar:
                x1, x2 = self.to_device([x1, x2])

                self.opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(enabled=self.amp, device_type=self.device):
                    z1 = self.model(x1)
                    z2 = self.model(x2)
                    loss = nt_xent_loss(z1, z2, temperature=self.temperature)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                losses.append(loss.item())
                pbar.set_postfix(loss=float(loss.item()))

            avg = sum(losses) / max(1, len(losses))
            print(f"[pretrain] ep{ep}: loss={avg:.4f}")
            if avg < best:
                best = avg
                save_ckpt(ckpt_path, self.model, extra={"best_loss": best})
        print(f"[pretrain] best_loss={best:.4f} saved: {ckpt_path}")
