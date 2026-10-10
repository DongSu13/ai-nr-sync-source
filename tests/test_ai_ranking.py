import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from ai_ranking import FEATURE_NAMES, split_by_seed, train_and_evaluate


def _record(seed: int, label: int, correlation: float) -> dict:
    return {
        "split_seed": seed,
        "candidate_source": "synthetic_test",
        "label": label,
        "features": {
            "pss_metric": correlation / 2,
            "sss_metric": correlation / 2,
            "correlation_metric": correlation,
            "abs_cfo_hz": 100.0 if label else 450.0,
            "received_power": 1.0,
            "residual_power": 0.1,
        },
    }


def test_training_keeps_seed_groups_disjoint_and_excludes_crc():
    records = [
        _record(seed, label, 1.8 if label else 0.4)
        for seed in range(1, 6)
        for label in (0, 1)
    ]
    splits = split_by_seed(records)
    assert set(splits["train_seeds"]).isdisjoint(splits["validation_seeds"])
    assert set(splits["train_seeds"]).isdisjoint(splits["test_seeds"])

    outcome = train_and_evaluate(records)
    assert outcome["model"]["feature_names"] == list(FEATURE_NAMES)
    assert "pbch_crc_ok" not in outcome["model"]["feature_names"]
    assert all(0.0 <= score <= 1.0 for score in outcome["test_scores"])
    assert all(isinstance(weight, int) for weight in outcome["model"]["int8_weights"])
    assert set(outcome["evaluation"]) == {"ai", "fixed_correlation"}


def test_training_rejects_missing_feature():
    record = _record(1, 1, 1.8)
    del record["features"]["pss_metric"]
    records = [record] + [
        _record(seed, label, 1.8 if label else 0.4)
        for seed in range(2, 6)
        for label in (0, 1)
    ]
    with pytest.raises(ValueError, match="pss_metric"):
        train_and_evaluate(records)
