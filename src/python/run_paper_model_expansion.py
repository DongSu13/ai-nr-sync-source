"""Serial development and one-time confirmation for eight CFO proposal models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from paper_aligned_train import predict_variant
from paper_model_expansion_data import build_partition, load_partition
from paper_model_expansion_models import MODEL_NAMES
from paper_model_expansion_train import predict_one, score_proposals, select_winner, train_one
from paper_repro_signal import grid_search, half_auto_search
from pss_cfo_ai import correlate_after_cfo, peak_hit


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("paper_model_expansion_data.py", "paper_model_expansion_models.py",
                "paper_model_expansion_train.py", "run_paper_model_expansion.py",
                "pss_cfo_ai.py", "paper_repro_signal.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(Path(__file__).parent / name) for name in SOURCE_FILES}


def _public(score: dict) -> dict:
    return {key: value for key, value in score.items()
            if key not in ("hit", "pred_offset", "pred_nid2", "cfo_squared_error")}


def run_development(output: Path, *, smoke: bool = False,
                    frozen_weight: Path | None = None,
                    resume: bool = False) -> dict:
    output = Path(output)
    if frozen_weight is None:
        raise ValueError("pass --reference-weight with a generated baseline checkpoint")
    frozen_weight = Path(frozen_weight)
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and not output.exists():
        raise FileNotFoundError(output)
    if (output / "selection.json").exists():
        raise FileExistsError(output / "selection.json")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    if not frozen_weight.is_file():
        raise FileNotFoundError(frozen_weight)
    if not resume:
        output.mkdir(parents=True)
    fit_count, validation_count = (4, 3) if smoke else (48_000, 12_000)
    data_root = output / "data"
    if not resume:
        fit_manifest = build_partition(data_root, "fit", 2_100_000, fit_count)
        val_manifest = build_partition(data_root, "validation", 2_148_000,
                                       validation_count)
    else:
        fit_manifest = json.loads((data_root / "fit" / "manifest.json").read_text())
        val_manifest = json.loads((data_root / "validation" / "manifest.json").read_text())
        if (fit_manifest["count"], val_manifest["count"]) != (fit_count, validation_count):
            raise ValueError("resume data counts differ from fixed run configuration")
    fit = load_partition(data_root, "fit")
    validation = load_partition(data_root, "validation")
    if np.intersect1d(fit["seeds"], validation["seeds"]).size:
        raise ValueError("fit/validation seed overlap")
    reference_proposal = predict_variant(frozen_weight, validation["cross"], False)["cfo_sc"]
    reference_score = score_proposals(validation["rx"], validation,
                                      reference_proposal)
    np.savez_compressed(output / "frozen_validation.npz", seeds=validation["seeds"],
                        cfo_sc=reference_proposal, hit=reference_score["hit"])
    records = {}
    weight_hashes = {}
    for name in MODEL_NAMES:
        weight = output / "weights" / f"{name}.pt"
        record_file = weight.with_suffix(".json")
        prediction_file = weight.with_suffix(".validation.npz")
        if resume and weight.exists() and record_file.exists() and prediction_file.exists():
            record = json.loads(record_file.read_text(encoding="utf-8"))
        elif weight.exists() or record_file.exists() or prediction_file.exists():
            raise ValueError(f"incomplete model checkpoint: {name}")
        else:
            record = train_one(name, fit, validation, weight,
                               max_epochs=1 if smoke else 60)
        records[name] = record
        weight_hashes[name] = _sha256(weight)
        print(f"EXPANSION MODEL COMPLETE {name}", flush=True)
    scored = {name: {**record["validation"],
                     "parameter_count": record["parameter_count"]}
              for name, record in records.items()}
    winner = select_winner(scored, reference_score)
    selection = {"winner": winner, "primary": "pss_hit_rate",
                 "secondary": "cfo_rmse_scs", "cfo_limit_factor": 1.05,
                 "frozen_reference": _public(reference_score),
                 "model_validation": scored,
                 "model_weight_sha256": weight_hashes,
                 "frozen_weight_path": str(frozen_weight.resolve()),
                 "frozen_weight_sha256": _sha256(frozen_weight),
                 "source_sha256": _source_hashes(),
                 "seed_ranges": {"fit": [2_100_000, 2_100_000 + fit_count - 1],
                                 "validation": [2_148_000,
                                                2_148_000 + validation_count - 1],
                                 "confirm": [2_200_000,
                                             2_200_002 if smoke else 2_202_999]},
                 "smoke": smoke}
    development = {"models": records, "selection": selection,
                   "partitions": {"fit": fit_manifest,
                                  "validation": val_manifest}}
    (output / "development.json").write_text(
        json.dumps(development, indent=2), encoding="utf-8")
    (output / "selection.json").write_text(json.dumps(selection, indent=2),
                                           encoding="utf-8")
    return development


def _candidate_result(rx: np.ndarray, truth: dict, proposal: float,
                      proposal_seconds: float = 0.) -> dict:
    began = time.perf_counter()
    estimate = correlate_after_cfo(rx, proposal)
    return {"cfo_sc": float(proposal), "offset": int(estimate["offset"]),
            "nid2": int(estimate["nid2"]), "score": float(estimate["score"]),
            "hit": bool(peak_hit(estimate, truth)),
            "cfo_squared_error": float((proposal - truth["cfo_sc"]) ** 2),
            "search_count": 1, "search_seconds": time.perf_counter() - began,
            "proposal_seconds": proposal_seconds}


def _grid_result(rx: np.ndarray, truth: dict) -> dict:
    began = time.perf_counter()
    estimate = grid_search(rx, np.round(np.arange(0., 3.8 + 1e-6, .1), 6))
    seconds = time.perf_counter() - began
    proposal = float(estimate["cfo_sc"])
    return {"cfo_sc": proposal, "offset": int(estimate["offset"]),
            "nid2": int(estimate["nid2"]), "score": float(estimate["score"]),
            "hit": bool(peak_hit(estimate, truth)),
            "cfo_squared_error": float((proposal - truth["cfo_sc"]) ** 2),
            "search_count": 39, "search_seconds": seconds,
            "proposal_seconds": 0.}


def _method_metrics(rows: list[dict], name: str) -> dict:
    values = [row["methods"][name] for row in rows]
    return {"count": len(values), "pss_hits": int(sum(value["hit"] for value in values)),
            "pss_hit_rate": float(np.mean([value["hit"] for value in values])),
            "cfo_rmse_scs": float(np.sqrt(np.mean(
                [value["cfo_squared_error"] for value in values]))),
            "search_count": int(sum(value["search_count"] for value in values)),
            "search_seconds": float(sum(value["search_seconds"] for value in values)),
            "proposal_seconds": float(sum(value["proposal_seconds"] for value in values))}


def _paired(rows: list[dict], challenger: str, reference: str) -> dict:
    a = np.array([row["methods"][challenger]["hit"] for row in rows], float)
    b = np.array([row["methods"][reference]["hit"] for row in rows], float)
    a_sq = np.array([row["methods"][challenger]["cfo_squared_error"]
                     for row in rows], float)
    b_sq = np.array([row["methods"][reference]["cfo_squared_error"]
                     for row in rows], float)
    rng = np.random.default_rng(2026)
    sample = rng.integers(0, len(rows), size=(2000, len(rows)))
    return {"pss_hit_gain_pp": float(100 * (a - b).mean()),
            "pss_hit_gain_ci95_pp": np.quantile(
                100 * (a - b)[sample].mean(axis=1), [.025, .975]).tolist(),
            "cfo_mse_gain_scs2": float((b_sq - a_sq).mean()),
            "cfo_mse_gain_ci95_scs2": np.quantile(
                (b_sq - a_sq)[sample].mean(axis=1), [.025, .975]).tolist(),
            "challenger_only_hits": int(np.sum((a == 1) & (b == 0))),
            "reference_only_hits": int(np.sum((a == 0) & (b == 1)))}


def run_confirmation(output: Path, *, smoke: bool = False) -> dict:
    output = Path(output)
    selection_path = output / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    target = output / "confirmation"
    if target.exists():
        raise FileExistsError(target)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if set(selection.get("model_weight_sha256", {})) != set(MODEL_NAMES):
        raise ValueError("incomplete frozen model selection")
    if bool(selection["smoke"]) != smoke:
        raise ValueError("confirmation scale disagrees with frozen selection")
    if selection["source_sha256"] != _source_hashes():
        raise ValueError("source changed after model selection")
    frozen_weight = Path(selection["frozen_weight_path"])
    if _sha256(frozen_weight) != selection["frozen_weight_sha256"]:
        raise ValueError("frozen CNN weight changed")
    for name in MODEL_NAMES:
        weight = output / "weights" / f"{name}.pt"
        if _sha256(weight) != selection["model_weight_sha256"][name]:
            raise ValueError(f"model weight changed: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU required")
    count = 3 if smoke else 3000
    started = time.perf_counter()
    manifest = build_partition(output / "data", "confirm", 2_200_000, count)
    data = load_partition(output / "data", "confirm")
    proposals = {}
    inference_seconds = {}
    began = time.perf_counter()
    proposals["frozen_cross_cnn"] = predict_variant(frozen_weight, data["cross"], False)["cfo_sc"]
    inference_seconds["frozen_cross_cnn"] = time.perf_counter() - began
    for name in MODEL_NAMES:
        began = time.perf_counter()
        proposals[name] = predict_one(name, output / "weights" / f"{name}.pt", data)
        inference_seconds[name] = time.perf_counter() - began
    target.mkdir()
    rows = []
    with (target / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for i, seed in enumerate(data["seeds"]):
            rx = data["rx"][i]
            truth = {key: int(data[key][i]) if key in ("offset", "nid2")
                     else float(data[key][i]) for key in
                     ("offset", "nid2", "cfo_sc")}
            began = time.perf_counter()
            classic_cfo = half_auto_search(rx, np.array([.8, 2., 3.5]))["cfo_sc"]
            classic_seconds = time.perf_counter() - began
            methods = {"zero": _candidate_result(rx, truth, 0.),
                       "classic": _candidate_result(rx, truth, classic_cfo,
                                                    classic_seconds),
                       "frozen_cross_cnn": _candidate_result(
                           rx, truth, float(proposals["frozen_cross_cnn"][i])),
                       "grid": _grid_result(rx, truth)}
            for name in MODEL_NAMES:
                methods[name] = _candidate_result(rx, truth, float(proposals[name][i]))
            row = {"seed": int(seed), "truth": truth,
                   "snr_db": float(data["snr_db"][i]),
                   "cfo_center_sc": float(data["cfo_center_sc"][i]),
                   "speed_kmh": float(data["speed_kmh"][i]),
                   "delay_spread_ns": float(data["delay_spread_ns"][i]),
                   "methods": methods}
            rows.append(row)
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if (i + 1) % 100 == 0:
                print(f"EXPANSION CONFIRM {i + 1}/{count}", flush=True)
    names = ("zero", "classic", "frozen_cross_cnn", *MODEL_NAMES, "grid")
    summary = {"count": count, "first_seed": 2_200_000,
               "last_seed": 2_200_000 + count - 1,
               "selected_winner": selection["winner"],
               "selection_sha256": _sha256(selection_path),
               "predictions_sha256": _sha256(target / "predictions.jsonl"),
               "partition": manifest,
               "device": torch.cuda.get_device_name(0),
               "methods": {name: _method_metrics(rows, name) for name in names},
               "inference_seconds": inference_seconds,
               "paired": {}, "strata": {},
               "total_seconds": time.perf_counter() - started}
    if selection["winner"] is not None:
        summary["paired"] = {
            reference: _paired(rows, selection["winner"], reference)
            for reference in ("classic", "frozen_cross_cnn")}
    for field in ("cfo_center_sc", "speed_kmh", "delay_spread_ns"):
        summary["strata"][field] = {}
        for value in sorted({row[field] for row in rows}):
            subset = [row for row in rows if row[field] == value]
            summary["strata"][field][str(value)] = {
                name: _method_metrics(subset, name) for name in names}
    for label, low, high in (("-5:-2", -5, -2), ("-1:2", -1, 2), ("3:6", 3, 6)):
        subset = [row for row in rows if low <= row["snr_db"] <= high]
        if subset:
            summary["strata"].setdefault("snr_bin", {})[label] = {
                name: _method_metrics(subset, name) for name in names}
    (target / "summary.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("develop", "confirm"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-weight", type=Path,
                        help="baseline checkpoint generated by run_paper_aligned.py")
    args = parser.parse_args()
    if args.mode == "develop":
        if args.reference_weight is None:
            parser.error("develop requires --reference-weight")
        run_development(args.output, smoke=args.smoke, resume=args.resume,
                         frozen_weight=args.reference_weight)
    else:
        if args.resume:
            parser.error("--resume applies only to develop")
        run_confirmation(args.output, smoke=args.smoke)
