"""CNN and CNN-LSTM ablation for isolated PSS synchronization."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SyncNet(nn.Module):
    def __init__(self, temporal: bool = True, max_cfo_sc: float = 3.5):
        super().__init__()
        self.temporal = temporal
        self.max_cfo_sc = max_cfo_sc
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, (1, 5), padding=(0, 2)), nn.ReLU(),
            nn.Conv2d(16, 24, (1, 5), padding=(0, 2)), nn.ReLU(),
            nn.Conv2d(24, 32, (1, 3), padding=(0, 1)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 16)),
        )
        if temporal:
            self.encoder = nn.LSTM(32 * 16, 40, num_layers=2, batch_first=True)
            dim = 40
        else:
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 16, 128), nn.ReLU())
            dim = 128
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.cnn(x.permute(0, 3, 1, 2))
        if self.temporal:
            z = z.permute(0, 2, 1, 3).reshape(len(x), 4, 32 * 16)
            z, _ = self.encoder(z)
            z = z[:, -1]
        else:
            z = self.encoder(z)
        y = self.head(z)
        return {"offset": torch.sigmoid(y[:, 0]) * 100.,
                "cfo_sc": torch.sigmoid(y[:, 1]) * self.max_cfo_sc,
                "sector_logits": y[:, 2:]}


def multitask_loss(out: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    return (F.smooth_l1_loss(out["offset"] / 100., target["offset"] / 100.) * 10
            + F.smooth_l1_loss(out["cfo_sc"] / 3.5, target["cfo_sc"] / 3.5) * 5
            + F.cross_entropy(out["sector_logits"], target["nid2"]))
