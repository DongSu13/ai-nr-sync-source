"""Single-process, four-arm paper-aligned PSS synchronization experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from paper_aligned_data import load_partition, write_partition
from paper_aligned_train import predict_variant, train_variant


PARTITIONS = {"fit": (1_000_000, 432_000),
              "validation": (1_500_000, 144_000),
              "test": (1_700_000, 144_000),
              "curve": (1_900_000, 12_000)}
VARIANTS = {"cross_cnn": ("cross", False),
            "cross_lstm": ("cross", True),
            "combined_cnn": ("combined", False),
            "combined_lstm": ("combined", True)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def metrics(pred: dict, truth: dict, selector: np.ndarray | None = None) -> dict:
    selected = slice(None) if selector is None else selector
    n = len(truth["offset"][selected])
    if not n:
        raise ValueError("empty metric slice")
    return {"count": n,
            "timing_rmse_samples": float(np.sqrt(np.mean(
                (pred["offset"][selected] - truth["offset"][selected]) ** 2))),
            "cfo_rmse_scs": float(np.sqrt(np.mean(
                (pred["cfo_sc"][selected] - truth["cfo_sc"][selected]) ** 2))),
            "sector_accuracy": float(np.mean(
                pred["nid2"][selected] == truth["nid2"][selected]))}


def relative_improvements(cnn: dict, lstm: dict) -> dict:
    return {"timing_rmse_reduction_pct": float(100 * (1 - lstm["timing_rmse_samples"]
                                                      / cnn["timing_rmse_samples"])),
            "cfo_rmse_reduction_pct": float(100 * (1 - lstm["cfo_rmse_scs"]
                                                   / cnn["cfo_rmse_scs"])),
            "sector_accuracy_gain_pp": float(100 * (lstm["sector_accuracy"]
                                                    - cnn["sector_accuracy"]))}


def _paired_bootstrap(cnn: dict, lstm: dict, truth: dict, *, draws: int = 1000) -> dict:
    rng = np.random.default_rng(2026)
    n = len(truth["offset"])
    values = {key: [] for key in ("timing_rmse_reduction_pct",
                                 "cfo_rmse_reduction_pct", "sector_accuracy_gain_pp")}
    for _ in range(draws):
        indices = rng.integers(0, n, n)
        gain = relative_improvements(metrics(cnn, truth, indices),
                                     metrics(lstm, truth, indices))
        for key in values:
            values[key].append(gain[key])
    return {key: np.quantile(value, [.025, .975]).tolist()
            for key, value in values.items()}


def _save_predictions(path: Path, prediction: dict, truth: dict) -> None:
    np.savez_compressed(path, **{f"pred_{key}": value for key, value in prediction.items()},
                        **{f"truth_{key}": truth[key] for key in
                           ("seeds", "offset", "cfo_sc", "nid2", "snr_db",
                            "speed_kmh", "delay_spread_ns")})


def run_paper_aligned(output: Path, *, smoke: bool = False) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    output.mkdir(parents=True)
    (output / "weights").mkdir()
    (output / "predictions").mkdir()
    counts = {key: (16 if key != "curve" else 12) if smoke else count
              for key, (_, count) in PARTITIONS.items()}
    summary = {"config": {"smoke": smoke, "max_epochs": 1 if smoke else 30,
                          "batch_size": 1000, "seed": 2026,
                          "curve_scenes_per_snr": 1 if smoke else 1000},
               "device": torch.cuda.get_device_name(0),
               "partitions": {}, "models": {}, "curves": {}, "paired": {},
               "test_strata": {}}
    start = time.perf_counter()
    for name, (first_seed, _) in PARTITIONS.items():
        summary["partitions"][name] = write_partition(
            output / "data", name, first_seed, counts[name], curve=name == "curve")
        print(f"PAPER PARTITION COMPLETE {name}", flush=True)
    fit = load_partition(output / "data", "fit")
    validation = load_partition(output / "data", "validation")
    test = load_partition(output / "data", "test")
    curve = load_partition(output / "data", "curve")
    if any(set(fit["seeds"]) & set(part["seeds"])
           for part in (validation, test, curve)):
        raise ValueError("seed leakage")
    curve_predictions = {}
    for name, (feature, temporal) in VARIANTS.items():
        began = time.perf_counter()
        weight = output / "weights" / f"{name}.pt"
        training = train_variant(fit, validation, feature, temporal,
                                 max_epochs=summary["config"]["max_epochs"],
                                 output=weight)
        training_seconds = time.perf_counter() - began
        began = time.perf_counter()
        test_pred = predict_variant(weight, test[feature], temporal)
        curve_pred = predict_variant(weight, curve[feature], temporal)
        inference_seconds = time.perf_counter() - began
        test_file = output / "predictions" / f"{name}_test.npz"
        curve_file = output / "predictions" / f"{name}.npz"
        _save_predictions(test_file, test_pred, test)
        _save_predictions(curve_file, curve_pred, curve)
        summary["models"][name] = {"training": training,
                                    "test": metrics(test_pred, test),
                                    "weight_sha256": _sha256(weight),
                                    "test_prediction_sha256": _sha256(test_file),
                                    "curve_prediction_sha256": _sha256(curve_file),
                                    "training_seconds": training_seconds,
                                    "inference_seconds": inference_seconds}
        summary["curves"][name] = {}
        for snr in range(-5, 7):
            selected = curve["snr_db"] == snr
            if np.any(selected):
                summary["curves"][name][str(snr)] = metrics(curve_pred, curve, selected)
        summary["test_strata"][name] = {}
        for field in ("cfo_sc", "speed_kmh", "delay_spread_ns"):
            summary["test_strata"][name][field] = {
                str(float(value)): metrics(test_pred, test, test[field] == value)
                for value in np.unique(test[field])}
        curve_predictions[name] = curve_pred
        (output / "progress.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
        print(f"PAPER MODEL COMPLETE {name}", flush=True)
    for feature in ("cross", "combined"):
        cnn = f"{feature}_cnn"
        lstm = f"{feature}_lstm"
        selected = curve["snr_db"] == -5
        cnn_metrics = metrics(curve_predictions[cnn], curve, selected)
        lstm_metrics = metrics(curve_predictions[lstm], curve, selected)
        truth = {key: curve[key][selected] for key in
                 ("offset", "cfo_sc", "nid2")}
        summary["paired"][feature] = {
            "snr_minus5": relative_improvements(cnn_metrics, lstm_metrics),
            "bootstrap_ci95": _paired_bootstrap(
                {key: value[selected] for key, value in curve_predictions[cnn].items()},
                {key: value[selected] for key, value in curve_predictions[lstm].items()},
                truth)}
    summary["total_seconds"] = time.perf_counter() - start
    (output / "summary.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_paper_aligned(args.output, smoke=args.smoke)
