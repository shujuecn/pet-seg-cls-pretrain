import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNetEncoder


class EncoderClassifier(nn.Module):
    """Encoder + linear head for cat/dog classification."""

    def __init__(self, num_classes: int = 2, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.encoder = UNetEncoder(base_channels=base_channels, depth=depth)
        self.fc = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.encoder(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.fc(feat)


class ProjectionHead(nn.Module):
    """Projection head for SimCLR."""

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderSimCLR(nn.Module):
    """Encoder + projection head for self-supervised pretraining."""

    def __init__(self, base_channels: int = 32, depth: int = 4, proj_dim: int = 128):
        super().__init__()
        self.encoder = UNetEncoder(base_channels=base_channels, depth=depth)
        self.proj = ProjectionHead(self.encoder.out_channels, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.encoder(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        z = self.proj(feat)
        z = F.normalize(z, dim=1)
        return z
