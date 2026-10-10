"""Isolated AI CFO proposal followed by a common NR-PSS cross-correlation search."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_repro_model import SyncNet
from paper_repro_signal import corr_features, grid_search, half_auto_search, make_scene
from run_paper_repro import ROOT


def scene_from_seed(seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed + 9_000_001)
    nid2 = int(rng.integers(0, 3))
    offset = int(rng.integers(8, 91))
    cfo_center = float(rng.choice([.8, 2., 3.5]))
    cfo = cfo_center + float(rng.uniform(-.15, .15))
    snr = float(rng.integers(-5, 7))
    speed = float(rng.choice([0., 100.]))
    spread = float(rng.choice([100., 300.]))
    rx, meta = make_scene(seed, nid2, offset, cfo, snr,
                          profile="TDL-A", delay_spread_ns=spread,
                          speed_kmh=speed)
    meta["cfo_center_sc"] = cfo_center
    return rx, meta


def correlate_after_cfo(rx: np.ndarray, cfo_sc: float) -> dict:
    return grid_search(rx, np.array([cfo_sc], dtype=float))


def peak_hit(pred: dict, truth: dict) -> bool:
    return (int(pred["nid2"]) == int(truth["nid2"])
            and abs(int(pred["offset"]) - int(truth["offset"])) <= 2)


def build_dataset(count: int, first_seed: int) -> dict:
    if count <= 0:
        raise ValueError("dataset must not be empty")
    features = np.empty((count, 4, 96, 2), np.float32)
    observations = np.empty((count, 384), np.complex64)
    seeds = np.arange(first_seed, first_seed + count, dtype=np.int64)
    labels = {key: np.empty(count, dtype)
              for key, dtype in (("cfo_sc", np.float32), ("cfo_center_sc", np.float32),
                                 ("offset", np.int32), ("nid2", np.int32),
                                 ("snr_db", np.float32), ("speed_kmh", np.float32),
                                 ("delay_spread_ns", np.float32))}
    for i, seed in enumerate(seeds):
        rx, meta = scene_from_seed(int(seed))
        features[i] = corr_features(rx, combined=True)
        observations[i] = rx
        for key in labels:
            labels[key][i] = meta[key]
        if (i + 1) % 5000 == 0:
            print(f"DATASET seed={first_seed} {i + 1}/{count}", flush=True)
    return {"x": features, "rx": observations, "seeds": seeds, **labels}


def evaluate_scene(rx: np.ndarray, truth: dict, ai_cfo_sc: float) -> dict:
    began = time.perf_counter()
    classic_cfo = half_auto_search(rx, np.array([.8, 2.0, 3.5]))["cfo_sc"]
    classic_estimate_seconds = time.perf_counter() - began
    proposals = {"zero": 0., "classic": classic_cfo, "ai": float(ai_cfo_sc)}
    results = {}
    for name, cfo in proposals.items():
        began = time.perf_counter()
        pred = correlate_after_cfo(rx, cfo)
        results[name] = {"cfo_sc": cfo, "offset": pred["offset"],
                         "nid2": pred["nid2"], "score": pred["score"],
                         "hit": bool(peak_hit(pred, truth)),
                         "cfo_squared_error": float((cfo - truth["cfo_sc"]) ** 2),
                         "search_count": 1,
                         "seconds": time.perf_counter() - began
                         + (classic_estimate_seconds if name == "classic" else 0.)}
    grid = np.round(np.arange(0, 3.8 + .00001, .1), 6)
    began = time.perf_counter()
    pred = grid_search(rx, grid)
    cfo = pred["cfo_sc"]
    results["grid"] = {"cfo_sc": cfo, "offset": pred["offset"],
                       "nid2": pred["nid2"], "score": pred["score"],
                       "hit": bool(peak_hit(pred, truth)),
                       "cfo_squared_error": float((cfo - truth["cfo_sc"]) ** 2),
                       "search_count": len(grid),
                       "seconds": time.perf_counter() - began}
    return results


def summarize_rows(rows: list[dict], seed: int = 2026) -> dict:
    if not rows or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate-seed scene list")
    methods = {}
    for name in ("zero", "classic", "ai", "grid"):
        hits = np.asarray([r["methods"][name]["hit"] for r in rows], bool)
        sq = np.asarray([r["methods"][name]["cfo_squared_error"] for r in rows], float)
        methods[name] = {"pss_hits": int(hits.sum()), "pss_hit_rate": float(hits.mean()),
                         "cfo_rmse_scs": float(np.sqrt(sq.mean()))}
    c = np.asarray([r["methods"]["classic"]["hit"] for r in rows], bool)
    a = np.asarray([r["methods"]["ai"]["hit"] for r in rows], bool)
    hit_diff = a.astype(float) - c.astype(float)
    sq_diff = np.asarray([r["methods"]["classic"]["cfo_squared_error"]
                          - r["methods"]["ai"]["cfo_squared_error"] for r in rows], float)
    rng = np.random.default_rng(seed)
    hit_boot = np.empty(5000)
    sq_boot = np.empty(5000)
    for i in range(5000):
        indices = rng.integers(0, len(rows), len(rows))
        hit_boot[i] = 100 * hit_diff[indices].mean()
        sq_boot[i] = sq_diff[indices].mean()
    paired = {"pss_hit_rate_gain_pp": float(100 * hit_diff.mean()),
              "pss_hit_bootstrap_ci95_pp": np.quantile(hit_boot, [.025, .975]).tolist(),
              "ai_only_pss_hits": int(np.sum(a & ~c)),
              "classic_only_pss_hits": int(np.sum(c & ~a)),
              "cfo_mse_gain_scs2": float(sq_diff.mean()),
              "cfo_mse_gain_bootstrap_ci95_scs2": np.quantile(sq_boot, [.025, .975]).tolist()}
    strata = {}
    for key, choices in (("cfo_center_sc", (.8, 2.0, 3.5)),
                         ("snr_bin", ("-5:-2", "-1:2", "3:6"))):
        groups = {}
        for choice in choices:
            selected = [r for r in rows if (abs(r["cfo_center_sc"] - choice) < 1e-4
                      if key == "cfo_center_sc"
                      else (-5 <= r["snr_db"] <= -2 if choice == "-5:-2" else
                            -1 <= r["snr_db"] <= 2 if choice == "-1:2" else
                            3 <= r["snr_db"] <= 6))]
            if selected:
                groups[str(choice)] = {"count": len(selected),
                    "classic_hits": int(sum(r["methods"]["classic"]["hit"] for r in selected)),
                    "ai_hits": int(sum(r["methods"]["ai"]["hit"] for r in selected))}
        strata[key] = groups
    return {"count": len(rows), "methods": methods,
            "paired_ai_minus_classic": paired, "strata": strata}


def predict_cfo(model: SyncNet, features: np.ndarray) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(features), 512):
            x = torch.from_numpy(features[start:start + 512]).to("cuda:0")
            output.append(model(x)["cfo_sc"].cpu().numpy())
    return np.concatenate(output)


def run_experiment(output: Path, train_count: int = 60000,
                   val_count: int = 9000, test_count: int = 3000,
                   epochs: int = 30, initialize: bool = False,
                   initial_weights: Path | None = None) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not (0 < train_count <= 60000 and 0 < val_count <= 9000
            and 0 < test_count <= 3000 and epochs > 0):
        raise ValueError("invalid counts or epochs for frozen seed ranges")
    if initialize and initial_weights is None:
        raise ValueError("warm starts require an explicit initial_weights checkpoint")
    if initialize and not Path(initial_weights).is_file():
        raise FileNotFoundError(initial_weights)
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    output.mkdir(parents=True)
    began = time.perf_counter()
    train = build_dataset(train_count, 900000)
    val = build_dataset(val_count, 960000)
    test = build_dataset(test_count, 970000)
    data_seconds = time.perf_counter() - began
    print("DATA READY", train_count, val_count, test_count, flush=True)
    began = time.perf_counter()
    model, history = train_cfo(train, val, epochs=epochs, initialize=initialize,
                               initial_weights=initial_weights)
    train_seconds = time.perf_counter() - began
    torch.save(model.state_dict(), output / "model.pt")
    del train, val
    began = time.perf_counter()
    ai_cfo = predict_cfo(model, test["x"])
    ai_inference_seconds = time.perf_counter() - began
    rows = []
    began = time.perf_counter()
    predictions = output / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as stream:
        for i, seed in enumerate(test["seeds"]):
            truth = {"offset": int(test["offset"][i]), "nid2": int(test["nid2"][i]),
                     "cfo_sc": float(test["cfo_sc"][i])}
            methods = evaluate_scene(test["rx"][i], truth, float(ai_cfo[i]))
            row = {"seed": int(seed), "snr_db": float(test["snr_db"][i]),
                   "cfo_center_sc": float(test["cfo_center_sc"][i]),
                   "speed_kmh": float(test["speed_kmh"][i]),
                   "delay_spread_ns": float(test["delay_spread_ns"][i]),
                   "truth": truth, "methods": methods}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if (i + 1) % 100 == 0:
                print(f"EVALUATION {i + 1}/{test_count}", flush=True)
    evaluation_seconds = time.perf_counter() - began
    summary = summarize_rows(rows)
    summary.update({"config": {"train": train_count, "val": val_count,
                               "test": test_count, "epochs": epochs,
                               "warm_start": initialize,
                               "train_first_seed": 900000,
                               "val_first_seed": 960000,
                               "test_first_seed": 970000,
                               "cfo_centers_sc": [.8, 2.0, 3.5],
                               "cfo_jitter_sc": [-.15, .15],
                               "full_grid_sc": [0., 3.8, .1]},
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "python": platform.python_version(),
                    "best_val_rmse_scs": min(h["val_rmse_scs"] for h in history),
                    "history": history,
                    "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(0) / 2**20,
                    "seconds": {"data": data_seconds, "training": train_seconds,
                                "ai_inference": ai_inference_seconds,
                                "four_policy_evaluation": evaluation_seconds},
                    "source_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()})
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    summary["gpu_snapshot"] = gpu.stdout.strip()
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"methods": summary["methods"],
                      "paired": summary["paired_ai_minus_classic"]},
                     ensure_ascii=False), flush=True)
    return summary


def train_cfo(train: dict, val: dict, epochs: int,
              initialize: bool = False,
              initial_weights: Path | None = None) -> tuple[SyncNet, list[dict]]:
    if initialize and initial_weights is None:
        raise ValueError("warm starts require an explicit initial_weights checkpoint")
    if initialize and not Path(initial_weights).is_file():
        raise FileNotFoundError(initial_weights)
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model = SyncNet(temporal=True, max_cfo_sc=3.8).to("cuda:0")
    if initialize:
        model.load_state_dict(torch.load(initial_weights, map_location="cuda:0",
                                         weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=.0008 if initialize else .002)
    x_train = torch.from_numpy(train["x"]).to("cuda:0")
    y_train = torch.from_numpy(train["cfo_sc"]).to("cuda:0")
    x_val = torch.from_numpy(val["x"]).to("cuda:0")
    y_val = torch.from_numpy(val["cfo_sc"]).to("cuda:0")
    history = []
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), device="cuda:0")
        losses = []
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(x_train[indices])["cfo_sc"],
                                    y_train[indices], beta=.25)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            squared = []
            for start in range(0, len(x_val), 512):
                error = model(x_val[start:start + 512])["cfo_sc"] - y_val[start:start + 512]
                squared.append(error.square())
            val_rmse = float(torch.sqrt(torch.cat(squared).mean()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_rmse_scs": val_rmse})
        print(f"CFO EPOCH {epoch + 1}/{epochs} train={np.mean(losses):.5f} "
              f"val_rmse={val_rmse:.5f}", flush=True)
        if val_rmse < best - .0001:
            best, bad = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 7:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "pss_cfo")
    parser.add_argument("--train", type=int, default=60000)
    parser.add_argument("--val", type=int, default=9000)
    parser.add_argument("--test", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=30)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--initial-weights", type=Path,
                         help="optional checkpoint generated with the training code")
    initial.add_argument("--no-initialize", action="store_true",
                         help="train from scratch (the default)")
    args = parser.parse_args()
    run_experiment(args.output, args.train, args.val, args.test,
                   args.epochs, args.initial_weights is not None, args.initial_weights)
