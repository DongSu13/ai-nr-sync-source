import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from paper_aligned_train import paper_mse_loss, predict_variant, train_variant


def test_paper_mse_loss_is_zero_for_exact_three_head_prediction():
    output = {"offset": torch.tensor([50.]), "cfo_sc": torch.tensor([1.75]),
              "sector_logits": torch.tensor([[1., 0., 0.]])}
    labels = {"offset": torch.tensor([50.]), "cfo_sc": torch.tensor([1.75]),
              "nid2": torch.tensor([0])}
    assert paper_mse_loss(output, labels).item() == 0.


def test_four_model_interface_trains_on_local_gpu_and_refuses_overwrite(tmp_path):
    assert torch.cuda.is_available()
    rng = np.random.default_rng(4)
    train = {"cross": rng.normal(size=(16, 4, 96, 2)).astype(np.float32),
             "combined": rng.normal(size=(16, 4, 96, 2)).astype(np.float32),
             "offset": np.linspace(8, 90, 16).astype(np.float32),
             "cfo_sc": np.resize(np.array([.8, 2., 3.5], np.float32), 16),
             "nid2": np.resize(np.array([0, 1, 2], np.int64), 16)}
    val = {key: array[:8] for key, array in train.items()}
    weights = tmp_path / "cross_lstm.pt"
    summary = train_variant(train, val, "cross", True,
                            max_epochs=1, output=weights)
    assert weights.is_file()
    assert summary["epochs"] == 1
    assert np.isfinite(summary["best_val_loss"])
    predicted = predict_variant(weights, val["cross"], temporal=True)
    assert predicted["cfo_sc"].shape == (8,)
    assert predicted["nid2"].shape == (8,)
    with pytest.raises(FileExistsError):
        train_variant(train, val, "cross", True,
                      max_epochs=1, output=weights)
