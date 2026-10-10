"""Budgeted user-recovery and candidate-level metrics for NR v3."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def budget_counts(scores: np.ndarray, d: dict, budget: int) -> np.ndarray:
    if budget < 0 or len(scores) != len(d["y"]):
        raise ValueError("invalid budget or score length")
    rows = []
    for seed in d["scene_ids"]:
        idx = np.flatnonzero(d["groups"] == seed)
        # mergesort preserves candidate ID order for equal scores.
        ordered = idx[np.argsort(-np.asarray(scores)[idx], kind="mergesort")]
        selected, seen = [], set()
        for j in ordered:
            if len(selected) >= budget:
                break
            pci = int(d["pcis"][j])
            if pci not in seen:
                seen.add(pci)
                selected.append(j)
        recovered = len({int(d["pcis"][j]) for j in selected if d["y"][j]})
        weak_recovered = len({int(d["pcis"][j]) for j in selected if d["y"][j] and d["weak"][j]})
        available = len({int(d["pcis"][j]) for j in idx if d["y"][j]})
        transmitted, weak_transmitted = d["true_users"][int(seed)], d["weak_users"][int(seed)]
        rows.append([recovered, transmitted, weak_recovered, weak_transmitted,
                     int(transmitted > 0 and recovered == transmitted), len(selected), available])
    return np.asarray(rows, dtype=int).reshape((-1, 7))


def _ratio(numerator: int, denominator: int):
    return None if denominator == 0 else numerator / denominator


def summarize_counts(counts: np.ndarray) -> dict:
    if np.asarray(counts).ndim != 2 or counts.shape[1] != 7:
        raise ValueError("counts must have seven columns")
    totals = counts.sum(axis=0)
    nonzero = counts[:, 1] > 0
    return {"recovered_users": int(totals[0]), "transmitted_users": int(totals[1]),
            "recovery_rate": _ratio(int(totals[0]), int(totals[1])),
            "weak_recovery_rate": _ratio(int(totals[2]), int(totals[3])),
            "full_recovery_rate": _ratio(int(counts[nonzero, 4].sum()), int(nonzero.sum())),
            "decode_attempts": int(totals[5]), "available_confirmable_users": int(totals[6]),
            "recovery_of_available": _ratio(int(totals[0]), int(totals[6])),
            "available_coverage": _ratio(int(totals[6]), int(totals[1]))}


def paired_user_ci(ai_counts: np.ndarray, base_counts: np.ndarray, reps: int = 5000, seed: int = 95341) -> dict:
    ai, base = np.asarray(ai_counts), np.asarray(base_counts)
    if ai.shape != base.shape or ai.ndim != 2 or ai.shape[1] != 7 or not np.array_equal(ai[:, 1], base[:, 1]):
        raise ValueError("paired counts require shared denominators")
    n, denominator = len(ai), int(ai[:, 1].sum())
    point = None if denominator == 0 else (ai[:, 0].sum() - base[:, 0].sum()) / denominator
    undefined = 0
    if n == 0 or reps <= 0 or denominator == 0:
        interval = [None, None]
    else:
        rng, deltas = np.random.default_rng(seed), []
        for _ in range(reps):
            sampled = rng.integers(0, n, n)
            den = ai[sampled, 1].sum()
            if den:
                deltas.append((ai[sampled, 0].sum() - base[sampled, 0].sum()) / den)
            else:
                undefined += 1
        interval = [float(x) for x in np.quantile(deltas, [.025, .975])] if deltas else [None, None]
    return {"delta_recovery": None if point is None else float(point), "ci95": interval, "scenes": int(n),
            "repetitions": reps, "undefined_replicates": undefined,
            "zero_denominator_policy": "omit undefined bootstrap draws, never impute zero"}


def threshold_at_fpr(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=int)
    if not 0 <= target <= 1 or len(scores) != len(y) or not np.isfinite(scores).all() or set(np.unique(y)) - {0, 1}:
        raise ValueError("invalid threshold inputs")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("threshold FPR requires both classes")
    # Score groups are atomic: cumulative totals are evaluated only at each tie boundary.
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores, ordered_y = scores[order], y[order]
    ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    cumulative_tp = np.cumsum(ordered_y)[ends]
    cumulative_fp = (ends + 1) - cumulative_tp
    thresholds = ordered_scores[ends]
    negatives = int((y == 0).sum())
    candidates = [(0, 0, float(np.nextafter(scores.max(), np.inf)))]
    candidates.extend((int(tp), int(fp), float(threshold)) for tp, fp, threshold in
                      zip(cumulative_tp, cumulative_fp, thresholds) if fp / negatives <= target)
    return max(candidates, key=lambda item: (item[0], -item[1], item[2]))[2]


def candidate_metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=int)
    if len(scores) != len(y) or not np.isfinite(scores).all() or set(np.unique(y)) - {0, 1}:
        raise ValueError("invalid candidate inputs")
    predicted = scores >= threshold
    tp, fp = int(((predicted) & (y == 1)).sum()), int(((predicted) & (y == 0)).sum())
    tn, fn = int(((~predicted) & (y == 0)).sum()), int(((~predicted) & (y == 1)).sum())
    positive, negative = tp + fn, fp + tn
    both = positive > 0 and negative > 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": _ratio(tp, positive), "fpr": _ratio(fp, negative),
            "AP": None if positive == 0 else float(average_precision_score(y, scores)),
            "AUC": None if not both else float(roc_auc_score(y, scores))}
