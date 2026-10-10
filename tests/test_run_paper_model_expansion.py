import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
import pss_cfo_ai
import run_paper_model_expansion as expansion
from paper_model_expansion_models import MODEL_NAMES
from paper_repro_model import SyncNet
from run_paper_model_expansion import run_confirmation, run_development


def test_development_requires_an_explicit_reference_checkpoint(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="reference-weight"):
        run_development(output, smoke=True)
    assert not output.exists()


def test_development_freezes_eight_models_before_fresh_confirmation(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen.pt"
    torch.save(SyncNet(temporal=False).state_dict(), frozen)
    output = tmp_path / "campaign"
    development = run_development(output, smoke=True, frozen_weight=frozen)
    assert set(development["models"]) == set(MODEL_NAMES)
    assert (output / "selection.json").is_file()
    assert development["selection"]["cfo_limit_factor"] == 1.05
    with pytest.raises(FileExistsError):
        run_development(output, smoke=True, frozen_weight=frozen)
    actual_correlate = pss_cfo_ai.correlate_after_cfo
    searches = []

    def counted_correlate(rx, proposal):
        searches.append(float(proposal))
        return actual_correlate(rx, proposal)

    monkeypatch.setattr(pss_cfo_ai, "correlate_after_cfo", counted_correlate)
    monkeypatch.setattr(expansion, "correlate_after_cfo", counted_correlate)
    confirmation = run_confirmation(output, smoke=True)
    rows = [json.loads(line) for line in (output / "confirmation" /
            "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["seed"] for row in rows] == [2_200_000, 2_200_001, 2_200_002]
    expected = {"zero", "classic", "frozen_cross_cnn", "grid", *MODEL_NAMES}
    assert all(set(row["methods"]) == expected for row in rows)
    assert all(row["methods"]["grid"]["search_count"] == 39 for row in rows)
    assert all(row["methods"]["cross_mlp"]["search_count"] == 1 for row in rows)
    assert len(searches) == 3 * (3 + len(MODEL_NAMES))
    assert all(row["methods"]["classic"]["proposal_seconds"] >= 0
               and row["methods"]["classic"]["search_seconds"] >= 0
               and row["methods"]["grid"]["search_seconds"] >= 0
               for row in rows)
    assert confirmation["count"] == 3
    with pytest.raises(FileExistsError):
        run_confirmation(output, smoke=True)


def test_confirmation_refuses_missing_or_changed_selection(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        run_confirmation(output, smoke=True)
