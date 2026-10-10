import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from run_paper_aligned import relative_improvements, run_paper_aligned


def test_relative_change_uses_cnn_as_denominator():
    cnn = {"timing_rmse_samples": 4., "cfo_rmse_scs": 2.,
           "sector_accuracy": .5}
    lstm = {"timing_rmse_samples": 2., "cfo_rmse_scs": 1.,
            "sector_accuracy": .6}
    result = relative_improvements(cnn, lstm)
    assert result["timing_rmse_reduction_pct"] == 50.
    assert result["cfo_rmse_reduction_pct"] == 50.
    assert result["sector_accuracy_gain_pp"] == pytest.approx(10.)


def test_smoke_runs_four_variants_on_shared_curves_and_refuses_overwrite(tmp_path):
    output = tmp_path / "paper"
    summary = run_paper_aligned(output, smoke=True)
    assert set(summary["models"]) == {"cross_cnn", "cross_lstm",
                                      "combined_cnn", "combined_lstm"}
    assert summary["partitions"]["fit"]["first_seed"] == 1_000_000
    assert summary["partitions"]["curve"]["first_seed"] == 1_900_000
    assert (output / "predictions" / "combined_lstm.npz").is_file()
    assert "-5" in summary["curves"]["combined_lstm"]
    assert (output / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        run_paper_aligned(output, smoke=True)
