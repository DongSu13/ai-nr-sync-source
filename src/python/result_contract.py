"""Validation for reproducible NR synchronization experiment results."""

from __future__ import annotations

from typing import Any


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field: {key}")
    return mapping[key]


def validate_result(payload: dict[str, Any], stage: str) -> None:
    """Raise ValueError unless a stage result has its minimum audit fields."""
    if payload.get("stage") != stage:
        raise ValueError(f"result stage must equal {stage}")
    _require(payload, "status", "result")
    _require(payload, "config", "result")
    metrics = _require(payload, "metrics", "result")

    if stage == "stage1":
        candidate = _require(payload, "candidate", "stage1 result")
        _require(candidate, "pci", "stage1 candidate")
        _require(candidate, "pbch_crc_ok", "stage1 candidate")
        _require(candidate, "timing_offset_samples", "stage1 candidate")
        _require(candidate, "cfo_hz", "stage1 candidate")
        _require(metrics, "pci_match", "stage1 metrics")
    elif stage == "stage2":
        candidates = _require(payload, "candidates", "stage2 result")
        if len(candidates) != 2:
            raise ValueError("stage2 result must contain exactly two candidates")
        _require(metrics, "pre_sic_power", "stage2 metrics")
        _require(metrics, "post_sic_residual_power", "stage2 metrics")
    else:
        raise ValueError(f"unknown stage: {stage}")
