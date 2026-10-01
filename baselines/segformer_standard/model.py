"""Standard SegFormer-B0 wrapper for repository-controlled benchmarks.

This baseline uses Hugging Face Transformers' SegFormer implementation and the
NVIDIA `nvidia/mit-b0` ImageNet-1K pretrained MiT-B0 encoder. The segmentation
head is initialized for the downstream class count and trained with the rest of
the model.

Inputs are expected to be float RGB tensors in [0, 1]. The wrapper applies the
ImageNet normalization used by the standard SegFormer preprocessing pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardSegFormerB0(nn.Module):
    MODEL_ID = "nvidia/mit-b0"

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from transformers import (
                SegformerConfig,
                SegformerForSemanticSegmentation,
                SegformerModel,
            )
        except ImportError as exc:
            raise ImportError(
                "Standard SegFormer-B0 requires the 'transformers' package. "
                "Install it with: pip install 'transformers>=4.46,<5'"
            ) from exc

        config = SegformerConfig(
            num_channels=3,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.1,
            classifier_dropout_prob=0.1,
            decoder_hidden_size=256,
            num_labels=num_classes,
            semantic_loss_ignore_index=-1,
        )
        self.model = SegformerForSemanticSegmentation(config)

        if pretrained:
            encoder = SegformerModel.from_pretrained(self.MODEL_ID)
            self.model.segformer.load_state_dict(encoder.state_dict(), strict=True)

        # Equivalent to mean/std [123.675, 116.28, 103.53] /
        # [58.395, 57.12, 57.375] on 0..255 RGB images.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)
        self.pretrained = bool(pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Standard SegFormer-B0 expects Bx3xHxW RGB input, got {tuple(x.shape)}"
            )

        height, width = x.shape[-2:]
        x = (x - self.pixel_mean) / self.pixel_std
        logits = self.model(pixel_values=x).logits
        return F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
