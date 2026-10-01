from __future__ import annotations

from torch import nn

from baselines.segformer.model import SegFormer
from baselines.segformer_standard import StandardSegFormerB0
from baselines.unet.model import UNet


SUPPORTED_MODELS = ("unet", "segformer", "segformer_b0")


def build_model(
    name: str,
    num_classes: int,
    in_channels: int = 3,
    mode: str = "full",
    variant: str | None = None,
    pretrained: bool = False,
) -> nn.Module:
    """Build a model while keeping smoke mode intentionally lightweight."""

    name = name.lower()
    if name == "unet":
        base = 8 if mode == "smoke" else 32
        return UNet(in_channels=in_channels, num_classes=num_classes, base=base)

    if name == "segformer":
        use_tiny = mode == "smoke" or variant == "tiny"
        if use_tiny:
            return SegFormer.tiny(in_channels=in_channels, num_classes=num_classes)

        # Self-contained repository implementation for development / ablation.
        return SegFormer(
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dims=(32, 64, 160, 256),
            depths=(2, 2, 2, 2),
            num_heads=(1, 2, 5, 8),
            sr_ratios=(8, 4, 2, 1),
        )

    if name == "segformer_b0":
        if in_channels != 3:
            raise ValueError(
                "Standard pretrained SegFormer-B0 currently supports RGB input only "
                "(--in-channels 3)."
            )
        return StandardSegFormerB0(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(SUPPORTED_MODELS)}")
