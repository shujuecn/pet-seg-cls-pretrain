import torch
from torch import nn


class ConvBlock(nn.Module):
    """两层卷积，每层都是 Conv + BN + ReLU。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UNetEncoder(nn.Module):
    """UNet Encoder 部分，显式定义每一层。"""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

    def forward(self, x: torch.Tensor):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        x5 = self.bottleneck(self.pool4(x4))
        return x1, x2, x3, x4, x5


class UNet(nn.Module):
    """完整 UNet，用于分割。"""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)

        self.up4_trans = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.up4_conv = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3_trans = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up3_conv = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2_trans = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up2_conv = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1_trans = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up1_conv = ConvBlock(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)

        u4 = self.up4_trans(x5)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.up4_conv(u4)

        u3 = self.up3_trans(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up3_conv(u3)

        u2 = self.up2_trans(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2_conv(u2)

        u1 = self.up1_trans(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up1_conv(u1)

        return self.out_conv(u1)


class EncoderClassifier(nn.Module):
    """复用 UNet Encoder，并接一个分类头。"""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, x5 = self.encoder(x)
        x = self.avg_pool(x5)
        x = torch.flatten(x, 1)
        return self.fc(x)
