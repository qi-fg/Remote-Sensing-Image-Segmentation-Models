"""A compact, self-contained SegFormer (Xie et al., NeurIPS 2021).

Implemented in pure PyTorch so the offline `--demo` runs **without** downloading
pretrained weights or needing `timm`/`mmseg`. The architecture follows the paper:

    Overlap Patch Embedding -> [Efficient Self-Attention + Mix-FFN] x N
    -> light-weight all-MLP decoder

Default config = a *tiny* variant (fast on CPU). Use `SegFormer.b0()`-style
channel widths via the constructor args for a closer-to-paper model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                       # B, C, H, W
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)       # B, N, C
        return self.norm(x), h, w


class EfficientSelfAttention(nn.Module):
    """Self-attention with spatial-reduction (sr_ratio) on K/V for efficiency."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        head_dim = c // self.num_heads
        q = self.q(x).reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    """Feed-forward with a depth-wise 3x3 conv between the two linear layers."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio), dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dims: tuple = (32, 64, 160, 256),
        depths: tuple = (2, 2, 2, 2),
        num_heads: tuple = (1, 2, 5, 8),
        sr_ratios: tuple = (8, 4, 2, 1),
        patch_sizes: tuple = (7, 3, 3, 3),
        strides: tuple = (4, 2, 2, 2),
        decoder_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_stages = 4
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj = nn.ModuleList()          # decoder per-stage linear projection

        in_ch = in_channels
        for i in range(self.num_stages):
            self.patch_embeds.append(OverlapPatchEmbed(in_ch, embed_dims[i],
                                                       patch_sizes[i], strides[i]))
            self.blocks.append(nn.ModuleList([
                Block(embed_dims[i], num_heads[i], sr_ratios[i]) for _ in range(depths[i])
            ]))
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            self.proj.append(nn.Conv2d(embed_dims[i], decoder_dim, kernel_size=1))
            in_ch = embed_dims[i]

        self.fuse = nn.Conv2d(decoder_dim * self.num_stages, decoder_dim, kernel_size=1)
        self.head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    @staticmethod
    def tiny(in_channels: int = 3, num_classes: int = 2) -> "SegFormer":
        """A small config that trains in seconds on CPU (for --demo)."""
        return SegFormer(in_channels, num_classes,
                         embed_dims=(16, 32, 64, 128), depths=(1, 1, 1, 1),
                         num_heads=(1, 2, 4, 8), sr_ratios=(8, 4, 2, 1),
                         decoder_dim=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2:]
        feats = []
        for i in range(self.num_stages):
            x, h, w = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, h, w)
            x = self.norms[i](x)
            b, _, c = x.shape
            x = x.transpose(1, 2).reshape(b, c, h, w)
            feats.append(x)

        target = feats[0].shape[-2:]
        outs = [F.interpolate(p(f), size=target, mode="bilinear", align_corners=False)
                for f, p in zip(feats, self.proj)]
        y = self.fuse(torch.cat(outs, dim=1))
        y = self.head(y)
        return F.interpolate(y, size=(in_h, in_w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    net = SegFormer.tiny()
    for h, w in [(64, 64), (128, 96)]:
        y = net(torch.randn(2, 3, h, w))
        assert y.shape == (2, 2, h, w), y.shape
        print(f"input (2,3,{h},{w}) -> output {tuple(y.shape)}  OK")
