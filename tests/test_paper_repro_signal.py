import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from paper_repro_signal import pss, make_scene, corr_features, grid_search, correlation_maps, half_auto_search


def test_pss_matches_nr_initial_register_and_three_distinct_sequences():
    # Catches a wrong NR register/shift convention, not merely a length change.
    seqs = [pss(i) for i in range(3)]
    np.testing.assert_array_equal(seqs[0][:7], [1, -1, -1, 1, -1, -1, -1])
    assert all(len(s) == 127 and set(s) == {-1, 1} for s in seqs)
    assert len({tuple(s) for s in seqs}) == 3


def test_clean_waveform_grid_search_recovers_time_cfo_and_sector():
    rx, truth = make_scene(seed=11, nid2=2, offset=41, cfo_sc=2.0, snr_db=80,
                           profile="AWGN")
    found = grid_search(rx, cfo_grid=np.arange(0, 3.51, 0.25))
    assert found["nid2"] == truth["nid2"]
    assert abs(found["offset"] - truth["offset"]) <= 1
    assert abs(found["cfo_sc"] - truth["cfo_sc"]) <= 0.25


def test_feature_shape_is_paper_table_and_finite():
    rx, _ = make_scene(seed=12, nid2=1, offset=40, cfo_sc=3.5, snr_db=-5,
                       profile="TDL-A")
    feat = corr_features(rx)
    assert feat.shape == (4, 96, 2)
    assert np.isfinite(feat).all()


def test_correlation_autocorrelation_feature_is_not_identical_to_plain_correlation():
    # Catches an invalid 128-sample lag applied to a truncated correlation vector.
    rx, _ = make_scene(seed=13, nid2=1, offset=40, cfo_sc=3.5, snr_db=-5,
                       profile="TDL-A")
    cross = corr_features(rx, combined=False)
    combined = corr_features(rx, combined=True)
    assert np.linalg.norm(cross - combined) > 0.01


def test_paper_half_symbol_autocorrelation_localizes_large_cfo():
    # Catches replacing equation (24) with a lag on an already truncated map.
    rx, _ = make_scene(seed=16, nid2=2, offset=38, cfo_sc=3.5, snr_db=80,
                       profile="AWGN")
    cross, half_auto = correlation_maps(rx)
    sector, offset = np.unravel_index(np.argmax(np.abs(half_auto)), half_auto.shape)
    assert cross.shape == half_auto.shape == (3, 111)
    assert sector == 2
    assert abs(offset - 38) <= 1


def test_half_auto_classic_recovers_clean_paper_cfo_grid():
    rx, _ = make_scene(seed=18, nid2=2, offset=38, cfo_sc=3.5, snr_db=80,
                       profile="AWGN")
    found = half_auto_search(rx, np.array([.8, 2., 3.5]))
    assert found["nid2"] == 2
    assert abs(found["offset"] - 38) <= 1
    assert found["cfo_sc"] == 3.5


def test_scene_repeats_exactly_for_same_seed():
    a, ta = make_scene(seed=15, nid2=0, offset=55, cfo_sc=0.8, snr_db=-3,
                       profile="TDL-A")
    b, tb = make_scene(seed=15, nid2=0, offset=55, cfo_sc=0.8, snr_db=-3,
                       profile="TDL-A")
    np.testing.assert_array_equal(a, b)
    assert ta == tb
