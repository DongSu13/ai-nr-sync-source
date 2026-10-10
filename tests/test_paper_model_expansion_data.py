import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from paper_model_expansion_data import branch_features, build_partition, load_partition
from pss_cfo_ai import scene_from_seed


def test_branch_features_preserve_three_pss_lag_axes_without_labels():
    rx, _ = scene_from_seed(2_100_000)
    cross = branch_features(rx, combined=False)
    combined = branch_features(rx, combined=True)
    assert cross.shape == (3, 111, 2)
    assert combined.shape == (3, 111, 4)
    assert np.array_equal(cross, branch_features(rx.copy(), combined=False))
    assert np.array_equal(combined[:, :, :2], cross)
    assert np.isfinite(combined).all()


def test_partition_uses_locked_seeds_labels_and_refuses_overwrite(tmp_path):
    root = tmp_path / "data"
    manifest = build_partition(root, "fit", 2_100_000, 3)
    data = load_partition(root, "fit")
    assert data["seeds"].tolist() == [2_100_000, 2_100_001, 2_100_002]
    assert data["cross"].shape == (3, 4, 96, 2)
    assert data["branch_cross"].shape == (3, 3, 111, 2)
    assert data["branch_combined"].shape == (3, 3, 111, 4)
    rx, truth = scene_from_seed(2_100_000)
    assert np.array_equal(data["rx"][0], rx)
    assert data["cfo_sc"][0] == pytest.approx(truth["cfo_sc"], abs=1e-6)
    assert manifest["count"] == 3
    assert len(manifest["sha256"]["rx.npy"]) == 64
    assert manifest["seconds"]["scene_generation"] > 0
    assert manifest["seconds"]["feature_extraction"] > 0
    with pytest.raises(FileExistsError):
        build_partition(root, "fit", 2_100_000, 3)


def test_partition_rejects_seed_window_leakage(tmp_path):
    with pytest.raises(ValueError):
        build_partition(tmp_path, "fit", 2_148_000, 1)
    with pytest.raises(ValueError):
        build_partition(tmp_path, "confirm", 2_200_000, 3_001)
