"""Single-process local-GPU PSS synchronization reproduction campaign."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from paper_repro_model import SyncNet, multitask_loss
from paper_repro_signal import corr_features, grid_search, half_auto_search, make_scene

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "paper_repro_v1"


def build_dataset(count: int, first_seed: int, combined: bool = True) -> dict:
    x = np.empty((count, 4, 96, 2), np.float32)
    x_alt = np.empty_like(x)
    labels = {"offset": np.empty(count, np.float32), "cfo_sc": np.empty(count, np.float32),
              "nid2": np.empty(count, np.int64), "snr_db": np.empty(count, np.float32),
              "speed_kmh": np.empty(count, np.float32), "spread_ns": np.empty(count, np.float32)}
    seeds = np.arange(first_seed, first_seed + count)
    observations = np.empty((count, 384), np.complex64)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed) + 9_000_001)
        nid2 = int(rng.integers(0, 3))
        offset = int(rng.integers(8, 91))
        cfo = float(rng.choice([.8, 2., 3.5]))
        snr = float(rng.integers(-5, 7))
        speed = float(rng.choice([0., 100.]))
        spread = float(rng.choice([100., 300.]))
        rx, _ = make_scene(int(seed), nid2, offset, cfo, snr,
                           profile="TDL-A", delay_spread_ns=spread, speed_kmh=speed)
        observations[i] = rx
        x[i] = corr_features(rx, combined=combined)
        x_alt[i] = corr_features(rx, combined=not combined)
        for key, value in (("offset", offset), ("cfo_sc", cfo), ("nid2", nid2),
                           ("snr_db", snr), ("speed_kmh", speed), ("spread_ns", spread)):
            labels[key][i] = value
    return {"x": x, "x_alt": x_alt, "rx": observations, "seeds": seeds, **labels}


def score_predictions(truth: dict, pred: dict) -> dict:
    a = np.asarray(truth["offset"], float) - np.asarray(pred["offset"], float)
    b = np.asarray(truth["cfo_sc"], float) - np.asarray(pred["cfo_sc"], float)
    c = np.asarray(truth["nid2"]) == np.asarray(pred["nid2"])
    return {"timing_mae_samples": float(np.mean(np.abs(a))),
            "timing_within_2_samples": float(np.mean(np.abs(a) <= 2)),
            "cfo_rmse_scs": float(np.sqrt(np.mean(b ** 2))),
            "cfo_within_025_scs": float(np.mean(np.abs(b) <= .25)),
            "cfo_within_05_scs": float(np.mean(np.abs(b) <= .5)),
            "sector_accuracy": float(np.mean(c))}


def _tensor_data(data: dict, feature: str, device: torch.device) -> dict:
    return {"x": torch.from_numpy(data[feature]).to(device),
            "offset": torch.from_numpy(data["offset"]).to(device),
            "cfo_sc": torch.from_numpy(data["cfo_sc"]).to(device),
            "nid2": torch.from_numpy(data["nid2"]).to(device)}


def train_model(train: dict, val: dict, feature: str, temporal: bool,
                max_epochs: int, seed: int = 2026) -> tuple[SyncNet, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU is required")
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    model = SyncNet(temporal).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.25, patience=3)
    tr = _tensor_data(train, feature, device)
    va = _tensor_data(val, feature, device)
    best = float("inf")
    best_state = None
    bad = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(tr["x"]), device=device)
        losses = []
        for batch in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            out = model(tr["x"][batch])
            target = {k: tr[k][batch] for k in ("offset", "cfo_sc", "nid2")}
            loss = multitask_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(multitask_loss(model(va["x"]), va))
        scheduler.step(val_loss)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)),
                            val_loss=val_loss, lr=optimizer.param_groups[0]["lr"]))
        print(f"epoch {epoch+1} train={np.mean(losses):.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best, "epochs": len(history), "history": history}


def predict_model(model: SyncNet, data: dict, feature: str) -> dict:
    model.eval()
    device = next(model.parameters()).device
    outputs = {"offset": [], "cfo_sc": [], "nid2": []}
    with torch.no_grad():
        for start in range(0, len(data[feature]), 512):
            x = torch.from_numpy(data[feature][start:start + 512]).to(device)
            out = model(x)
            outputs["offset"].extend(out["offset"].cpu().numpy().tolist())
            outputs["cfo_sc"].extend(out["cfo_sc"].cpu().numpy().tolist())
            outputs["nid2"].extend(out["sector_logits"].argmax(1).cpu().numpy().tolist())
    return {k: np.asarray(v) for k, v in outputs.items()}


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    began = time.time()
    print("generating isolated TDL-A scenes", flush=True)
    train = build_dataset(args.train, 100_000, combined=True)
    val = build_dataset(args.val, 200_000, combined=True)
    test = build_dataset(args.test, 300_000, combined=True)
    configs = [("cnn_cross", "x_alt", False), ("cnn_combined", "x", False),
               ("cnnlstm_combined", "x", True)]
    truth = {k: test[k] for k in ("offset", "cfo_sc", "nid2")}
    summary = {"config": vars(args), "device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "python": platform.python_version(), "models": {}, "classical": {}}
    prediction_arrays = {f"truth_{k}": v for k, v in truth.items()}
    prediction_arrays["test_seeds"] = test["seeds"]
    prediction_arrays["test_snr_db"] = test["snr_db"]
    for name, feature, temporal in configs:
        print("training", name, "on", summary["device"], flush=True)
        model, fit = train_model(train, val, feature, temporal, args.epochs)
        pred = predict_model(model, test, feature)
        torch.save(model.state_dict(), out / f"{name}.pt")
        summary["models"][name] = {"fit": fit, "overall": score_predictions(truth, pred),
            "low_snr": score_predictions({k: v[test["snr_db"] <= -2] for k, v in truth.items()},
                {k: v[test["snr_db"] <= -2] for k, v in pred.items()})}
        for k, v in pred.items():
            prediction_arrays[f"{name}_{k}"] = v
        print(name, summary["models"][name]["overall"], flush=True)
    for name, grid in (("zero_cfo", np.array([0.])),
                       ("cfo_grid", np.arange(0, 3.501, .25)),
                       ("paper_half_auto", np.array([.8, 2., 3.5]))):
        print("evaluating classic", name, flush=True)
        rows = ([half_auto_search(rx, grid) for rx in test["rx"]]
                if name == "paper_half_auto" else [grid_search(rx, grid) for rx in test["rx"]])
        pred = {k: np.array([r[k] for r in rows]) for k in truth}
        summary["classical"][name] = score_predictions(truth, pred)
        for k, v in pred.items():
            prediction_arrays[f"{name}_{k}"] = v
    summary["elapsed_seconds"] = time.time() - began
    summary["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(0) / 2**20
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    summary["gpu_snapshot"] = gpu.stdout.strip()
    np.savez_compressed(out / "predictions.npz", **prediction_arrays)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--val", type=int, default=900)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    run(parser.parse_args())
