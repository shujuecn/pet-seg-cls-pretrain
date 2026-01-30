import torch.nn as nn
import torch.nn.functional as F
from .unet import UNetEncoder

class EncoderClassifier(nn.Module):
    def __init__(self, num_classes=2, base_channels=32, depth=4):
        super().__init__()
        self.encoder = UNetEncoder(base_channels=base_channels, depth=depth)
        self.fc = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x):
        feat, _ = self.encoder(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.fc(feat)

class ProjectionHead(nn.Module):
    """自监督对比学习投影头"""
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )
    def forward(self, x): return self.net(x)

class EncoderSimCLR(nn.Module):
    def __init__(self, base_channels=32, depth=4, proj_dim=128):
        super().__init__()
        self.encoder = UNetEncoder(base_channels=base_channels, depth=depth)
        self.proj = ProjectionHead(self.encoder.out_channels, proj_dim)

    def forward(self, x):
        import torch.nn.functional as F
        feat, _ = self.encoder(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        z = self.proj(feat)
        z = F.normalize(z, dim=1)
        return z
