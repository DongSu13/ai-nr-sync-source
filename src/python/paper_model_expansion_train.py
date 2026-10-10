"""Common one-GPU training and PSS-first selection for CFO proposals."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_model_expansion_models import create_model, feature_for
from pss_cfo_ai import correlate_after_cfo, peak_hit


def epoch_better(hits: int, cfo_rmse: float,
                 best_hits: int, best_cfo_rmse: float) -> bool:
    return hits > best_hits or (hits == best_hits and cfo_rmse < best_cfo_rmse)


def select_winner(records: dict, frozen_reference: dict) -> str | None:
    limit = 1.05 * float(frozen_reference["cfo_rmse_scs"])
    eligible = [(name, value) for name, value in records.items()
                if float(value["cfo_rmse_scs"]) <= limit + 1e-9]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (
        -float(item[1]["pss_hit_rate"]), float(item[1]["cfo_rmse_scs"]),
        int(item[1]["parameter_count"]), item[0]))[0]


def score_proposals(rx: np.ndarray, truth: dict,
                    proposed_cfo: np.ndarray) -> dict:
    proposals = np.asarray(proposed_cfo, dtype=np.float64).reshape(-1)
    if len(rx) != len(proposals) or any(len(truth[key]) != len(proposals)
                                       for key in ("offset", "nid2", "cfo_sc")):
        raise ValueError("paired scene and prediction lengths disagree")
    if not len(proposals):
        raise ValueError("empty scene list")
    hits = np.empty(len(proposals), bool)
    offsets = np.empty(len(proposals), np.int16)
    nid2s = np.empty(len(proposals), np.int8)
    began = time.perf_counter()
    for i, cfo in enumerate(proposals):
        estimated = correlate_after_cfo(rx[i], float(cfo))
        offsets[i], nid2s[i] = estimated["offset"], estimated["nid2"]
        scene_truth = {key: truth[key][i] for key in ("offset", "nid2", "cfo_sc")}
        hits[i] = peak_hit(estimated, scene_truth)
    squared = (proposals - np.asarray(truth["cfo_sc"], float)) ** 2
    return {"count": len(proposals), "pss_hits": int(hits.sum()),
            "pss_hit_rate": float(hits.mean()),
            "cfo_rmse_scs": float(np.sqrt(squared.mean())),
            "search_count": len(proposals),
            "search_seconds": time.perf_counter() - began,
            "hit": hits, "pred_offset": offsets, "pred_nid2": nid2s,
            "cfo_squared_error": squared}


def _public_score(score: dict) -> dict:
    return {key: value for key, value in score.items()
            if key not in ("hit", "pred_offset", "pred_nid2", "cfo_squared_error")}


def _predict_model(model: torch.nn.Module, array: np.ndarray) -> tuple[np.ndarray, float]:
    model.eval()
    predictions = []
    began = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(array), 512):
            batch = torch.from_numpy(np.array(array[start:start + 512],
                                              dtype=np.float32, copy=True)).to("cuda:0")
            predictions.append(model(batch)["cfo_sc"].cpu().numpy())
    return np.concatenate(predictions), time.perf_counter() - began


def predict_one(name: str, weights: Path, data: dict) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    model = create_model(name).to("cuda:0")
    model.load_state_dict(torch.load(weights, map_location="cuda:0", weights_only=True))
    predictions, _ = _predict_model(model, data[feature_for(name)])
    return predictions


def train_one(name: str, fit: dict, validation: dict, output: Path,
              *, max_epochs: int = 60) -> dict:
    output = Path(output)
    history_path = output.with_suffix(".json")
    predictions_path = output.with_suffix(".validation.npz")
    if output.exists() or history_path.exists() or predictions_path.exists():
        raise FileExistsError(output)
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if np.intersect1d(fit["seeds"], validation["seeds"]).size:
        raise ValueError("fit/validation seed leakage")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    feature = feature_for(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(2026)
    model = create_model(name).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3)
    torch.cuda.reset_peak_memory_stats(0)
    history = []
    best_hits, best_cfo, best_epoch, bad = -1, float("inf"), 0, 0
    best_state = None
    best_prediction = None
    best_score = None
    training_seconds = validation_inference_seconds = validation_search_seconds = 0.
    for epoch in range(max_epochs):
        model.train()
        began = time.perf_counter()
        losses = []
        order = rng.permutation(len(fit[feature]))
        for start in range(0, len(order), 512):
            idx = order[start:start + 512]
            x = torch.from_numpy(np.array(fit[feature][idx], dtype=np.float32,
                                          copy=True)).to("cuda:0")
            target = torch.from_numpy(np.array(fit["cfo_sc"][idx],
                                               dtype=np.float32, copy=True)).to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            result = model(x)
            loss = F.smooth_l1_loss(result["cfo_sc"] / 3.8, target / 3.8,
                                    beta=.05)
            if "center_logits" in result:
                center = np.array(fit["cfo_center_sc"][idx], copy=True)
                classes = np.argmin(np.abs(center[:, None]
                                           - np.array([.8, 2., 3.5])[None]), axis=1)
                labels = torch.from_numpy(classes).long().to("cuda:0")
                loss = loss + .2 * F.cross_entropy(result["center_logits"], labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        training_seconds += time.perf_counter() - began
        proposed, inference_seconds = _predict_model(model, validation[feature])
        validation_inference_seconds += inference_seconds
        score = score_proposals(validation["rx"], validation, proposed)
        validation_search_seconds += score["search_seconds"]
        scheduler.step(score["cfo_rmse_scs"])
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "pss_hits": score["pss_hits"], "pss_hit_rate": score["pss_hit_rate"],
               "cfo_rmse_scs": score["cfo_rmse_scs"],
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"EXPANSION TRAIN {name} epoch={epoch + 1}/{max_epochs} "
              f"pss={score['pss_hits']}/{score['count']} "
              f"cfo={score['cfo_rmse_scs']:.5f}", flush=True)
        if epoch_better(score["pss_hits"], score["cfo_rmse_scs"],
                        best_hits, best_cfo):
            best_hits, best_cfo, best_epoch, bad = (
                score["pss_hits"], score["cfo_rmse_scs"], epoch + 1, 0)
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_prediction, best_score = proposed.copy(), score
        else:
            bad += 1
            if bad >= 10:
                break
    assert best_state is not None and best_score is not None
    torch.save(best_state, output)
    np.savez_compressed(predictions_path, seeds=validation["seeds"],
                        cfo_sc=best_prediction, hit=best_score["hit"],
                        pred_offset=best_score["pred_offset"],
                        pred_nid2=best_score["pred_nid2"])
    record = {"name": name, "feature": feature, "epochs": len(history),
              "best_epoch": best_epoch, "validation": _public_score(best_score),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "history": history, "device": torch.cuda.get_device_name(0),
              "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(0) / 2**20,
              "seconds": {"training": training_seconds,
                          "validation_inference": validation_inference_seconds,
                          "validation_search": validation_search_seconds}}
    history_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
