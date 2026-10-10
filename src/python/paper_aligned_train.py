"""Single-GPU, same-data CNN/CNN-LSTM training with normalized three-task MSE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_repro_model import SyncNet


def paper_mse_loss(output: dict[str, torch.Tensor],
                   labels: dict[str, torch.Tensor]) -> torch.Tensor:
    sector = F.one_hot(labels["nid2"].long(), num_classes=3).float()
    return (F.mse_loss(output["offset"] / 100., labels["offset"] / 100.)
            + F.mse_loss(output["cfo_sc"] / 3.5, labels["cfo_sc"] / 3.5)
            + F.mse_loss(output["sector_logits"], sector))


def _tensors(data: dict, feature: str, indices: np.ndarray | slice) -> tuple[torch.Tensor, dict]:
    x = torch.from_numpy(np.array(data[feature][indices], dtype=np.float32,
                                  copy=True)).to("cuda:0")
    labels = {key: torch.from_numpy(np.array(data[key][indices], copy=True)).to("cuda:0")
              for key in ("offset", "cfo_sc", "nid2")}
    return x, labels


def train_variant(train: dict, val: dict, feature: str, temporal: bool,
                  *, max_epochs: int, output: Path) -> dict:
    output = Path(output)
    history_path = output.with_suffix(".json")
    if output.exists() or history_path.exists():
        raise FileExistsError(output)
    if feature not in ("cross", "combined") or max_epochs <= 0:
        raise ValueError("invalid model variant or epoch budget")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    rng = np.random.default_rng(2026)
    model = SyncNet(temporal=temporal).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.25, patience=3)
    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for indices in np.array_split(rng.permutation(len(train[feature])),
                                      int(np.ceil(len(train[feature]) / 1000))):
            x, labels = _tensors(train, feature, indices)
            optimizer.zero_grad(set_to_none=True)
            loss = paper_mse_loss(model(x), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val_weighted = 0.
        with torch.no_grad():
            for start in range(0, len(val[feature]), 1000):
                stop = min(start + 1000, len(val[feature]))
                x, labels = _tensors(val, feature, slice(start, stop))
                val_weighted += float(paper_mse_loss(model(x), labels)) * (stop - start)
        val_loss = val_weighted / len(val[feature])
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        print(f"PAPER TRAIN {feature} {'lstm' if temporal else 'cnn'} "
              f"epoch={epoch + 1}/{max_epochs} val={val_loss:.6f}", flush=True)
        if val_loss < best - 1e-5:
            best, best_epoch, bad = val_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    assert best_state is not None
    torch.save(best_state, output)
    summary = {"feature": feature, "temporal": temporal,
               "epochs": len(history), "best_epoch": best_epoch,
               "best_val_loss": best, "history": history,
               "device": torch.cuda.get_device_name(0)}
    history_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return summary


def predict_variant(weights: Path, feature_array: np.ndarray,
                    temporal: bool) -> dict[str, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    model = SyncNet(temporal=temporal).to("cuda:0")
    model.load_state_dict(torch.load(weights, map_location="cuda:0", weights_only=True))
    model.eval()
    output = {"offset": [], "cfo_sc": [], "nid2": []}
    with torch.no_grad():
        for start in range(0, len(feature_array), 1000):
            x = torch.from_numpy(np.array(feature_array[start:start + 1000],
                                          dtype=np.float32, copy=True)).to("cuda:0")
            pred = model(x)
            output["offset"].append(pred["offset"].cpu().numpy())
            output["cfo_sc"].append(pred["cfo_sc"].cpu().numpy())
            output["nid2"].append(pred["sector_logits"].argmax(1).cpu().numpy())
    return {key: np.concatenate(value) for key, value in output.items()}
