import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from run_paper_repro import build_dataset, score_predictions


def test_dataset_splits_have_disjoint_seeds_and_public_shapes():
    # Catches train/test scene reuse or a feature/target misalignment.
    train = build_dataset(4, first_seed=100, combined=True)
    test = build_dataset(3, first_seed=200, combined=True)
    assert set(train["seeds"]).isdisjoint(test["seeds"])
    assert train["x"].shape == (4, 4, 96, 2)
    assert train["offset"].shape == (4,)
    assert set(train["nid2"]).issubset({0, 1, 2})
    assert np.isfinite(train["x"]).all()


def test_metrics_exact_for_hand_checked_predictions():
    # Catches a Celsius-like units mixup between samples and SCS units.
    truth = {"offset": np.array([10., 20.]), "cfo_sc": np.array([.8, 2.]),
             "nid2": np.array([0, 2])}
    pred = {"offset": np.array([11., 18.]), "cfo_sc": np.array([1., 1.5]),
            "nid2": np.array([0, 1])}
    got = score_predictions(truth, pred)
    assert got["timing_mae_samples"] == 1.5
    assert abs(got["cfo_rmse_scs"] - np.sqrt((.2 ** 2 + .5 ** 2) / 2)) < 1e-12
    assert got["sector_accuracy"] == .5
