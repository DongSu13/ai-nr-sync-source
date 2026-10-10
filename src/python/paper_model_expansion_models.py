"""Eight fixed CFO-proposal architectures for the PSS model comparison."""
from __future__ import annotations

import torch
from torch import nn

from paper_repro_model import SyncNet


MODEL_NAMES = ("cross_cnn_refit", "cross_mlp", "cross_resnet", "cross_tcn",
               "cross_attention", "branch_direct_cross",
               "branch_coarse_residual_cross",
               "branch_coarse_residual_combined")


def feature_for(name: str) -> str:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {name}")
    if name == "branch_coarse_residual_combined":
        return "branch_combined"
    if name.startswith("branch_"):
        return "branch_cross"
    return "cross"


class MlpCfo(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(768, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> dict:
        return {"cfo_sc": torch.sigmoid(self.net(x).squeeze(-1)) * 3.8}


def _cross_sequence(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 1, 3, 2).reshape(len(x), 8, 96)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation,
                      dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class ConvCfo(nn.Module):
    def __init__(self, dilations: tuple[int, ...]):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(8, 48, 5, padding=2), nn.ReLU(),
                                     *(ResidualBlock(48, d) for d in dilations),
                                     nn.AdaptiveAvgPool1d(8), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(48 * 8, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(_cross_sequence(x))
        return {"cfo_sc": torch.sigmoid(self.head(z).squeeze(-1)) * 3.8}


class AttentionCfo(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 64)
        self.position = nn.Parameter(torch.zeros(1, 96, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> dict:
        tokens = _cross_sequence(x).transpose(1, 2)
        z = self.encoder(self.embed(tokens) + self.position).mean(dim=1)
        return {"cfo_sc": torch.sigmoid(self.head(z).squeeze(-1)) * 3.8}


class BranchCfo(nn.Module):
    def __init__(self, channels: int, coarse_residual: bool):
        super().__init__()
        self.coarse_residual = coarse_residual
        self.shared = nn.Sequential(nn.Conv1d(channels, 32, 5, padding=2), nn.ReLU(),
                                    ResidualBlock(32), nn.Conv1d(32, 48, 3, padding=1),
                                    nn.ReLU(), nn.AdaptiveAvgPool1d(8))
        self.fuse = nn.Sequential(nn.Flatten(), nn.Linear(3 * 48 * 8, 160),
                                  nn.ReLU())
        self.head = nn.Linear(160, 6 if coarse_residual else 1)
        self.register_buffer("centers", torch.tensor([.8, 2., 3.5]))

    def forward(self, x: torch.Tensor) -> dict:
        batch, branches, lag, channels = x.shape
        if branches != 3 or lag != 111:
            raise ValueError("expected three 111-lag PSS branches")
        z = x.permute(0, 1, 3, 2).reshape(batch * 3, channels, lag)
        z = self.shared(z).reshape(batch, 3, 48, 8)
        y = self.head(self.fuse(z))
        if not self.coarse_residual:
            return {"cfo_sc": torch.sigmoid(y[:, 0]) * 3.8}
        logits = y[:, :3]
        residual = .15 * torch.tanh(y[:, 3:])
        proposal = torch.sum(torch.softmax(logits, dim=1)
                             * (self.centers + residual), dim=1)
        return {"cfo_sc": proposal.clamp(0., 3.8), "center_logits": logits}


def create_model(name: str) -> nn.Module:
    if name == "cross_cnn_refit":
        return SyncNet(temporal=False, max_cfo_sc=3.8)
    if name == "cross_mlp":
        return MlpCfo()
    if name == "cross_resnet":
        return ConvCfo((1, 1, 1))
    if name == "cross_tcn":
        return ConvCfo((1, 2, 4, 8))
    if name == "cross_attention":
        return AttentionCfo()
    if name == "branch_direct_cross":
        return BranchCfo(2, False)
    if name == "branch_coarse_residual_cross":
        return BranchCfo(2, True)
    if name == "branch_coarse_residual_combined":
        return BranchCfo(4, True)
    raise ValueError(f"unknown model: {name}")
