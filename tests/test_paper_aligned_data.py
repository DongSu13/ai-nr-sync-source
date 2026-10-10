import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from paper_aligned_data import load_partition, paper_scene, write_partition


def test_same_seed_reproduces_paper_scene_and_discrete_targets():
    first, labels = paper_scene(1_000_001)
    second, repeated = paper_scene(1_000_001)
    assert np.array_equal(first, second)
    assert labels == repeated
    assert labels["cfo_sc"] in (.8, 2., 3.5)
    assert labels["nid2"] in (0, 1, 2)
    assert -5 <= labels["snr_db"] <= 6


def test_curve_scene_locks_paper_conditions_by_seed():
    _, first = paper_scene(1_900_000, curve=True)
    _, last = paper_scene(1_911_999, curve=True)
    assert (first["snr_db"], first["cfo_sc"], first["nid2"],
            first["speed_kmh"], first["delay_spread_ns"]) == (-5., 3.5, 2, 0., 100.)
    assert last["snr_db"] == 6.
    with pytest.raises(ValueError):
        paper_scene(1_912_000, curve=True)


def test_partition_persists_two_feature_views_and_refuses_overwrite(tmp_path):
    write_partition(tmp_path, "fit", 1_000_000, 3)
    data = load_partition(tmp_path, "fit")
    assert data["cross"].shape == (3, 4, 96, 2)
    assert data["combined"].shape == (3, 4, 96, 2)
    assert data["seeds"].tolist() == [1_000_000, 1_000_001, 1_000_002]
    assert data["cfo_sc"].shape == (3,)
    original = np.asarray(data["cross"]).copy()
    with pytest.raises(FileExistsError):
        write_partition(tmp_path, "fit", 1_000_000, 3)
    assert np.array_equal(load_partition(tmp_path, "fit")["cross"], original)
