from __future__ import annotations

import unittest

import torch
from torch import nn

from train import build_optimizer, build_scheduler


class _DummySegFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.encoder = nn.Linear(4, 4)
        self.model.layer_norm = nn.LayerNorm(4)
        self.model.decode_head = nn.Linear(4, 2)


class TrainingRecipeTest(unittest.TestCase):
    def _cfg(self) -> dict:
        return {
            "optimizer_profile": "segformer_official",
            "lr": 6e-5,
            "weight_decay": 1e-2,
            "head_lr_mult": 10.0,
            "scheduler": "poly",
            "warmup_iters": 2,
            "warmup_ratio": 1e-2,
            "poly_power": 1.0,
        }

    def test_segformer_optimizer_head_multiplier_and_norm_no_decay(self) -> None:
        model = _DummySegFormer()
        optimizer = build_optimizer(model, self._cfg())

        group_settings = {
            (round(group["lr"], 10), group["weight_decay"])
            for group in optimizer.param_groups
        }
        self.assertIn((0.00006, 0.01), group_settings)
        self.assertIn((0.0006, 0.01), group_settings)
        self.assertIn((0.00006, 0.0), group_settings)

    def test_poly_scheduler_warmup_then_decays(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        cfg = self._cfg()
        scheduler = build_scheduler(optimizer, cfg, total_iters=10)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01, places=6)
        values = []
        for _ in range(4):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])

        self.assertGreater(values[1], values[0])
        self.assertLess(values[-1], values[1])


if __name__ == "__main__":
    unittest.main()
