import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from result_contract import validate_result


def test_stage1_requires_candidate_identity_and_pbch_confirmation():
    payload = {
        "stage": "stage1",
        "status": "success",
        "config": {"ncellid": 42},
        "candidate": {"pci": 42},
        "metrics": {},
    }

    with pytest.raises(ValueError, match="pbch_crc_ok"):
        validate_result(payload, "stage1")


def test_stage2_requires_residual_power_measurement():
    payload = {
        "stage": "stage2",
        "status": "success",
        "config": {},
        "candidates": [{"pci": 1}, {"pci": 2}],
        "metrics": {"pre_sic_power": 1.0},
    }

    with pytest.raises(ValueError, match="post_sic_residual_power"):
        validate_result(payload, "stage2")
