import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from pss_cfo_ai import (build_dataset, correlate_after_cfo, evaluate_scene,
                        peak_hit, run_experiment, scene_from_seed,
                        summarize_rows, train_cfo)
from paper_repro_signal import make_scene


def test_scene_seed_reproduces_continuous_cfo_and_labels():
    first, meta = scene_from_seed(900123)
    second, repeated = scene_from_seed(900123)
    assert np.array_equal(first, second)
    assert meta == repeated
    assert .65 <= meta["cfo_sc"] <= 3.65
    assert -5 <= meta["snr_db"] <= 6
    assert meta["nid2"] in (0, 1, 2)


def test_correct_peak_requires_both_sector_and_two_sample_timing():
    truth = {"offset": 30, "nid2": 1}
    assert peak_hit({"offset": 32, "nid2": 1}, truth)
    assert not peak_hit({"offset": 33, "nid2": 1}, truth)
    assert not peak_hit({"offset": 30, "nid2": 2}, truth)


def test_single_cfo_compensation_finds_clean_pss_peak():
    rx, truth = make_scene(23, nid2=2, offset=24, cfo_sc=2.0,
                           snr_db=60, profile="AWGN")
    pred = correlate_after_cfo(rx, 2.0)
    assert peak_hit(pred, truth)
    assert pred["cfo_sc"] == 2.0


def test_warm_start_requires_an_explicit_checkpoint_before_creating_outputs(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="initial_weights"):
        run_experiment(output, train_count=2, val_count=2, test_count=2,
                       epochs=1, initialize=True)
    assert not output.exists()


def test_training_rejects_an_unspecified_warm_start_checkpoint():
    with pytest.raises(ValueError, match="initial_weights"):
        train_cfo({}, {}, epochs=1, initialize=True)


def test_cfo_trainer_uses_local_gpu_and_returns_finite_validation_rmse():
    import torch
    assert torch.cuda.is_available()
    train = {"x": np.random.default_rng(1).standard_normal((16, 4, 96, 2)).astype(np.float32),
             "cfo_sc": np.linspace(.7, 3.6, 16).astype(np.float32)}
    val = {"x": np.random.default_rng(2).standard_normal((8, 4, 96, 2)).astype(np.float32),
           "cfo_sc": np.linspace(.7, 3.6, 8).astype(np.float32)}
    model, history = train_cfo(train, val, epochs=1, initialize=False)
    assert next(model.parameters()).is_cuda
    assert len(history) == 1
    assert np.isfinite(history[0]["val_rmse_scs"])


def test_dataset_records_disjoint_seeded_waveforms():
    train = build_dataset(2, 900000)
    val = build_dataset(2, 960000)
    assert train["seeds"].tolist() == [900000, 900001]
    assert val["seeds"].tolist() == [960000, 960001]
    assert train["x"].shape == (2, 4, 96, 2)
    assert train["rx"].shape == (2, 384)
    assert train["cfo_sc"].shape == (2,)


def test_all_pss_methods_share_peak_rule_and_report_search_budget():
    rx, truth = make_scene(23, nid2=2, offset=24, cfo_sc=2.0,
                           snr_db=60, profile="AWGN")
    result = evaluate_scene(rx, truth, ai_cfo_sc=2.0)
    assert set(result) == {"zero", "classic", "ai", "grid"}
    assert result["ai"]["hit"] is True
    assert result["ai"]["cfo_squared_error"] == 0
    assert result["ai"]["search_count"] == 1
    assert result["grid"]["search_count"] == 39
    assert result["grid"]["hit"] is True
    assert all(np.isfinite(result[name]["seconds"]) for name in result)


def test_summary_uses_paired_scene_differences_not_unpaired_rates():
    rows = [
        {"seed": 1, "snr_db": -5, "cfo_center_sc": .8,
         "methods": {"classic": {"hit": False, "cfo_squared_error": 1.0},
                     "ai": {"hit": True, "cfo_squared_error": .25},
                     "zero": {"hit": False, "cfo_squared_error": 1.0},
                     "grid": {"hit": True, "cfo_squared_error": 0.0}}},
        {"seed": 2, "snr_db": 5, "cfo_center_sc": 3.5,
         "methods": {"classic": {"hit": True, "cfo_squared_error": 0.0},
                     "ai": {"hit": True, "cfo_squared_error": .25},
                     "zero": {"hit": False, "cfo_squared_error": 4.0},
                     "grid": {"hit": True, "cfo_squared_error": 0.0}}},
    ]
    summary = summarize_rows(rows)
    assert summary["count"] == 2
    assert summary["methods"]["classic"]["pss_hit_rate"] == .5
    assert summary["methods"]["ai"]["pss_hit_rate"] == 1.0
    assert summary["paired_ai_minus_classic"]["pss_hit_rate_gain_pp"] == 50
    assert summary["paired_ai_minus_classic"]["ai_only_pss_hits"] == 1
    assert summary["paired_ai_minus_classic"]["classic_only_pss_hits"] == 0
    rows[0]["cfo_center_sc"] = float(np.float32(.8))
    assert summarize_rows(rows)["strata"]["cfo_center_sc"]["0.8"]["count"] == 1


def test_experiment_writes_auditable_rows_and_refuses_overwrite(tmp_path):
    import json
    import pytest
    output = tmp_path / "experiment"
    run_experiment(output, train_count=8, val_count=4, test_count=3,
                   epochs=1, initialize=False)
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [r["seed"] for r in rows] == [970000, 970001, 970002]
    assert summary["count"] == 3
    assert summary["source_sha256"]
    assert (output / "model.pt").is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, train_count=8, val_count=4, test_count=3,
                       epochs=1, initialize=False)
