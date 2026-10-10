import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from paper_model_expansion_models import MODEL_NAMES, create_model, feature_for


@pytest.mark.parametrize("name", [
    "cross_cnn_refit", "cross_mlp", "cross_resnet", "cross_tcn",
    "cross_attention", "branch_direct_cross",
    "branch_coarse_residual_cross", "branch_coarse_residual_combined"])
def test_every_model_proposes_one_bounded_cfo_with_gradient(name):
    model = create_model(name)
    shape = {"cross": (4, 96, 2), "branch_cross": (3, 111, 2),
             "branch_combined": (3, 111, 4)}[feature_for(name)]
    output = model(torch.randn(2, *shape))
    assert output["cfo_sc"].shape == (2,)
    assert torch.isfinite(output["cfo_sc"]).all()
    assert torch.all((output["cfo_sc"] >= 0) & (output["cfo_sc"] <= 3.8))
    if "coarse_residual" in name:
        assert output["center_logits"].shape == (2, 3)
    else:
        assert "center_logits" not in output
    output["cfo_sc"].sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_unknown_name_is_not_silently_mapped_to_a_model():
    assert len(MODEL_NAMES) == 8
    with pytest.raises(ValueError):
        create_model("unknown")
    with pytest.raises(ValueError):
        feature_for("unknown")
