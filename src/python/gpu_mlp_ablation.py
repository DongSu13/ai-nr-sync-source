"""GPU MLP ablation for the NR candidate-ranking dataset.

This is an experiment comparator, not the deployment model.  It consumes
exactly the same non-CRC features as the logistic-regression baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "pss_metric", "sss_metric", "correlation_metric", "abs_cfo_hz",
    "received_power", "residual_power",
)


def validate_feature_schema(records: list[dict]) -> None:
    prohibited = {"pbch_crc_ok", "truth_pci", "pci", "timing_error_samples", "cfo_error_hz"}
    for record in records:
        features = record.get("features", {})
        missing = set(FEATURE_NAMES) - set(features)
        if missing:
            raise ValueError(f"missing feature(s): {sorted(missing)}")
        if prohibited & set(features):
            raise ValueError("CRC or truth fields are prohibited model inputs")


def _metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "detection_rate": tp / (tp + fn) if tp + fn else 0.0,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0}


def _threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    eligible = []
    for value in np.unique(scores):
        metric = _metrics(scores, labels, float(value))
        if metric["false_alarm_rate"] <= 0.10:
            eligible.append((metric["detection_rate"], -metric["false_alarm_rate"], float(value)))
    return max(eligible)[2] if eligible else float(np.nextafter(np.max(scores), np.inf))


def run(data_path: Path, output_path: Path, seed: int) -> None:
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
    validate_feature_schema(records)
    seed_groups = sorted({int(record["split_seed"]) for record in records})
    train_seeds, validation_seeds, test_seeds = seed_groups[:3], seed_groups[3:4], seed_groups[4:]
    def select(seeds: list[int]):
        rows = [record for record in records if int(record["split_seed"]) in seeds]
        x = np.array([[row["features"][key] for key in FEATURE_NAMES] for row in rows], dtype=np.float32)
        y = np.array([row["label"] for row in rows], dtype=np.float32)
        return x, y
    x_train, y_train = select(train_seeds)
    x_validation, y_validation = select(validation_seeds)
    x_test, y_test = select(test_seeds)
    mean, std = x_train.mean(0), x_train.std(0)
    std[std == 0] = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    x = torch.tensor((x_train - mean) / std, device=device)
    y = torch.tensor(y_train[:, None], device=device)
    for _ in range(400):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
    def scores(array: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor((array - mean) / std, device=device)
            return torch.sigmoid(model(tensor)).squeeze(1).cpu().numpy()
    validation_scores, test_scores = scores(x_validation), scores(x_test)
    threshold = _threshold(validation_scores, y_validation.astype(int))
    result = {"seed": seed, "device": str(device), "feature_names": list(FEATURE_NAMES),
              "splits": {"train": train_seeds, "validation": validation_seeds, "test": test_seeds},
              "validation_threshold": threshold,
              "test": _metrics(test_scores, y_test.astype(int), threshold),
              "test_scores": test_scores.tolist(), "test_labels": y_test.astype(int).tolist()}
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.data, args.output, args.seed)
