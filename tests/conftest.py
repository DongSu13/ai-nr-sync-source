from __future__ import annotations

import pytest


CUDA_TESTS = {
    "test_cfo_trainer_uses_local_gpu_and_returns_finite_validation_rmse",
    "test_experiment_writes_auditable_rows_and_refuses_overwrite",
    "test_smoke_runs_four_variants_on_shared_curves_and_refuses_overwrite",
    "test_multitask_loss_backpropagates_on_local_gpu",
    "test_four_model_interface_trains_on_local_gpu_and_refuses_overwrite",
    "test_one_epoch_gpu_training_saves_frozen_prediction_and_refuses_overwrite",
    "test_development_freezes_eight_models_before_fresh_confirmation",
}


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA is required for this training smoke test")
    for item in items:
        if item.name in CUDA_TESTS:
            item.add_marker(skip_cuda)
