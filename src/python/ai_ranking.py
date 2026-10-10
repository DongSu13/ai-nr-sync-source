"""Train and evaluate a deployable AI reliability ranker for NR candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


FEATURE_NAMES = (
    "pss_metric",
    "sss_metric",
    "correlation_metric",
    "abs_cfo_hz",
    "received_power",
    "residual_power",
)


def _validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) < 5:
        raise ValueError("at least five seed-grouped records are required")
    for record in records:
        features = record.get("features", {})
        for feature in FEATURE_NAMES:
            if feature not in features:
                raise ValueError(f"missing required feature: {feature}")
        prohibited = {"pbch_crc_ok", "truth_pci", "pci", "timing_error_samples", "cfo_error_hz"}
        if prohibited.intersection(features):
            raise ValueError("label or truth information must not be a model feature")
        if record.get("label") not in (0, 1):
            raise ValueError("label must be binary")


def split_by_seed(records: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_records(records)
    seeds = sorted({int(record["split_seed"]) for record in records})
    if len(seeds) < 5:
        raise ValueError("at least five distinct seeds are required")
    train_count = max(3, int(np.floor(len(seeds) * 0.6)))
    validation_count = max(1, int(np.floor(len(seeds) * 0.2)))
    if train_count + validation_count >= len(seeds):
        validation_count = 1
        train_count = len(seeds) - 2
    return {
        "train_seeds": seeds[:train_count],
        "validation_seeds": seeds[train_count : train_count + validation_count],
        "test_seeds": seeds[train_count + validation_count :],
    }


def _matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(record["features"][name]) for name in FEATURE_NAMES] for record in records])
    y = np.array([int(record["label"]) for record in records])
    return x, y


def _records_for(records: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    allowed = set(seeds)
    return [record for record in records if int(record["split_seed"]) in allowed]


def _select_threshold(scores: np.ndarray, labels: np.ndarray, max_false_alarm: float = 0.10) -> float:
    if not ({0, 1} <= set(labels.tolist())):
        raise ValueError("validation data must contain both positive and negative labels")
    candidates = np.unique(scores)
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        metrics = _classification_metrics(scores, labels, float(threshold))
        key = (metrics["detection_rate"], -metrics["false_alarm_rate"], float(threshold))
        if metrics["false_alarm_rate"] <= max_false_alarm and (best is None or key > best):
            best = key
    if best is None:
        threshold = float(np.nextafter(np.max(scores), np.inf))
    else:
        threshold = best[2]
    return threshold


def _classification_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    true_negative = int(np.sum(~predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    return {
        "threshold": float(threshold),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "detection_rate": true_positive / positives if positives else 0.0,
        "false_alarm_rate": false_positive / negatives if negatives else 0.0,
        "average_precision": float(average_precision_score(labels, scores)),
    }


def train_and_evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    splits = split_by_seed(records)
    train_records = _records_for(records, splits["train_seeds"])
    validation_records = _records_for(records, splits["validation_seeds"])
    test_records = _records_for(records, splits["test_seeds"])
    x_train, y_train = _matrix(train_records)
    x_validation, y_validation = _matrix(validation_records)
    x_test, y_test = _matrix(test_records)
    if not ({0, 1} <= set(y_train.tolist())):
        raise ValueError("training data must contain both classes")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    classifier = LogisticRegression(
        class_weight="balanced", random_state=20260922, solver="liblinear", max_iter=1000
    )
    classifier.fit((x_train - mean) / std, y_train)
    validation_scores = classifier.predict_proba((x_validation - mean) / std)[:, 1]
    test_scores = classifier.predict_proba((x_test - mean) / std)[:, 1]
    ai_threshold = _select_threshold(validation_scores, y_validation)

    correlation_index = FEATURE_NAMES.index("correlation_metric")
    fixed_threshold = _select_threshold(x_validation[:, correlation_index], y_validation)
    ai_metrics = _classification_metrics(test_scores, y_test, ai_threshold)
    fixed_metrics = _classification_metrics(x_test[:, correlation_index], y_test, fixed_threshold)

    raw_weights = classifier.coef_[0]
    raw_intercept = float(classifier.intercept_[0])
    quantization_scale = max(float(np.max(np.abs(raw_weights))), abs(raw_intercept), 1e-12) / 127.0
    model = {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": raw_weights.tolist(),
        "intercept": raw_intercept,
        "int8_weights": np.rint(raw_weights / quantization_scale).astype(int).tolist(),
        "int8_intercept": int(np.rint(raw_intercept / quantization_scale)),
        "quantization_scale": quantization_scale,
    }
    return {
        "splits": splits,
        "model": model,
        "thresholds": {"ai": ai_threshold, "fixed_correlation": fixed_threshold},
        "evaluation": {"ai": ai_metrics, "fixed_correlation": fixed_metrics},
        "test_scores": test_scores.tolist(),
        "test_labels": y_test.tolist(),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_training_artifacts(records_path: Path, output_dir: Path) -> dict[str, Any]:
    outcome = train_and_evaluate(load_records(records_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.json").write_text(
        json.dumps(outcome["model"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outcome


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    write_training_artifacts(root / "outputs" / "ai" / "candidates.jsonl", root / "outputs" / "ai")
