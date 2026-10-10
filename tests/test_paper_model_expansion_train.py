import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from paper_model_expansion_data import build_partition, load_partition
from paper_model_expansion_train import (epoch_better, predict_one,
                                         score_proposals, select_winner, train_one)
from paper_repro_signal import make_scene


def test_score_proposals_uses_true_cfo_and_common_pss_hit_rule():
    rx, truth = make_scene(42, 1, 20, 2., 30., profile="AWGN")
    labels = {key: np.array([truth[key]]) for key in ("offset", "nid2", "cfo_sc")}
    score = score_proposals(rx[None], labels, np.array([2.]))
    assert score["pss_hits"] == 1
    assert score["pss_hit_rate"] == 1.
    assert score["cfo_rmse_scs"] == 0.
    assert score["search_count"] == 1


def test_epoch_and_model_selection_are_pss_first_with_cfo_constraint():
    assert epoch_better(91, .30, 90, .10)
    assert epoch_better(91, .29, 91, .30)
    assert not epoch_better(90, .01, 91, .30)
    records = {"a": {"pss_hit_rate": .91, "cfo_rmse_scs": .21,
                     "parameter_count": 100},
               "b": {"pss_hit_rate": .95, "cfo_rmse_scs": .211,
                     "parameter_count": 10},
               "c": {"pss_hit_rate": .89, "cfo_rmse_scs": .10,
                     "parameter_count": 10}}
    assert select_winner(records, {"cfo_rmse_scs": .20}) == "a"
    assert select_winner({"b": records["b"]}, {"cfo_rmse_scs": .20}) is None


def test_one_epoch_gpu_training_saves_frozen_prediction_and_refuses_overwrite(tmp_path):
    assert torch.cuda.is_available()
    root = tmp_path / "data"
    build_partition(root, "fit", 2_100_000, 4)
    build_partition(root, "validation", 2_148_000, 3)
    fit = load_partition(root, "fit")
    validation = load_partition(root, "validation")
    output = tmp_path / "cross_mlp.pt"
    record = train_one("cross_mlp", fit, validation, output, max_epochs=1)
    assert record["best_epoch"] == 1
    assert record["validation"]["search_count"] == 3
    assert output.is_file()
    proposals = predict_one("cross_mlp", output, validation)
    assert proposals.shape == (3,)
    assert np.isfinite(proposals).all()
    with pytest.raises(FileExistsError):
        train_one("cross_mlp", fit, validation, output, max_epochs=1)
