"""A minimal, dependency-free U-Net for remote sensing semantic segmentation.

Pure PyTorch (torch + torch.nn only). Works with any number of input bands
(e.g. RGB=3, multispectral/HSI=N) and any number of classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Classic U-Net (Ronneberger et al., MICCAI 2015).

    Args:
        in_channels: number of input bands.
        num_classes: number of output classes (2 for binary segmentation).
        base: base channel width (32 for the standard "small" U-Net).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        # Decoder (transposed conv + skip concat + double conv)
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c5, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)  # (B, num_classes, H, W) logits


if __name__ == "__main__":
    # Quick shape sanity check (no data / no GPU needed).
    for h, w in [(64, 64), (128, 96)]:
        net = UNet(in_channels=3, num_classes=2, base=8)
        y = net(torch.randn(2, 3, h, w))
        assert y.shape == (2, 2, h, w), y.shape
        print(f"input (2,3,{h},{w}) -> output {tuple(y.shape)}  OK")
