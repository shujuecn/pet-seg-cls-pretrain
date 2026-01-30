import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = ConvBNReLU(in_ch, out_ch)
        self.c2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.c1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.c2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, base_channels=32, depth=4, in_channels=3):
        super().__init__()
        self.depth = depth
        chs = [base_channels * (2**i) for i in range(depth)]
        self.stem = DownBlock(in_channels, chs[0])
        self.downs = nn.ModuleList(
            [DownBlock(chs[i - 1], chs[i]) for i in range(1, depth)]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DownBlock(chs[-1], chs[-1] * 2)
        self.out_channels = chs[-1] * 2
        self.skip_channels = chs

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for blk in self.downs:
            x = self.pool(x)
            x = blk(x)
            skips.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        return x, skips


class UNet(nn.Module):
    def __init__(self, num_classes=1, base_channels=32, depth=4, in_channels=3):
        super().__init__()
        self.encoder = UNetEncoder(base_channels, depth, in_channels)
        skip_chs = self.encoder.skip_channels
        cur = self.encoder.out_channels
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(UpBlock(cur, skip_chs[i], skip_chs[i]))
            cur = skip_chs[i]
        self.head = nn.Conv2d(cur, num_classes, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        for up, i in zip(self.ups, reversed(range(len(skips)))):
            x = up(x, skips[i])
        return self.head(x)
