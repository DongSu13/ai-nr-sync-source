"""Strict, leakage-free loading for NR v3 scene measurements."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FEATURES = (
    "pss", "sss", "correlation", "peak_ratio", "consistency", "abs_cfo",
    "dmrs_power", "dmrs_coherence", "dmrs_fit_error", "dmrs_variation",
    "llr_abs", "llr_low_fraction", "estimated_noise",
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _as_list(value, field: str) -> list:
    if value is None:
        _fail(f"{field} is required")
    return value if isinstance(value, list) else [value]


def _payload(value, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 24 or any(type(x) is not int for x in value):
        _fail(f"{field} must be a 24-bit integer list")
    if any(x not in (0, 1) for x in value):
        _fail(f"{field} must contain bits")
    return value


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{field} must be finite")
    return float(value)


def _exact_integer(value, field: str) -> int:
    # JSON has numbers, not a separate integer type; MATLAB may use exponents.
    number = _finite_number(value, field)
    if abs(value) > 2**53 or not number.is_integer():
        _fail(f"{field} must be an exact safe integer")
    return int(value)


def _validate_scene(raw: dict) -> dict:
    if not isinstance(raw, dict):
        _fail("scene must be an object")
    required = ("seed", "split", "suite", "cell", "config", "config_hash", "schema", "users",
                "candidates", "receiver_hash", "timings", "ref_power", "noise_variance")
    if any(key not in raw for key in required) or raw["schema"] != "nr_gain_v3":
        _fail("invalid nr_gain_v3 scene schema")
    seed = _exact_integer(raw["seed"], "seed")
    if not isinstance(raw["split"], str) or not isinstance(raw["config"], dict):
        _fail("invalid scene identity")
    for field in ("split", "suite", "cell", "config_hash", "receiver_hash"):
        if not isinstance(raw[field], str) or not raw[field]:
            _fail(f"{field} must be a nonempty string")
    if not isinstance(raw['timings'], dict):
        _fail('timings must be an object')
    for name, value in raw['timings'].items():
        if _finite_number(value, f'timings.{name}') < 0:
            _fail('negative timing')
    _finite_number(raw["ref_power"], "ref_power")
    _finite_number(raw["noise_variance"], "noise_variance")
    users = _as_list(raw["users"], "users") if raw["users"] != [] else []
    normalized_users = []
    seen_pcis = set()
    for i, user in enumerate(users):
        if not isinstance(user, dict) or type(user.get("pci")) is not int:
            _fail(f"users[{i}] malformed")
        pci = user["pci"]
        if pci in seen_pcis:
            _fail("unique user PCI required")
        seen_pcis.add(pci)
        normalized_users.append({**user, "channel_seed": _exact_integer(user.get('channel_seed'), 'channel_seed'),
                                 "payload": _payload(user.get("payload"), f"users[{i}].payload"),
                                 "power_db": _finite_number(user.get("power_db"), f"users[{i}].power_db")})
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 12:
        _fail("candidates must be a stable list of 12")
    payload_by_pci = {u["pci"]: u["payload"] for u in normalized_users}
    normalized_candidates = []
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or type(candidate.get("pci")) is not int or type(candidate.get("crc")) is not bool:
            _fail(f"candidates[{i}] malformed")
        features = candidate.get("features")
        if not isinstance(features, dict) or set(features) != set(FEATURES):
            _fail(f"candidates[{i}].features must be the exact feature whitelist")
        normalized_features = {name: _finite_number(features[name], f"candidates[{i}].features.{name}") for name in FEATURES}
        payload = _payload(candidate.get("payload"), f"candidates[{i}].payload")
        expected = candidate["crc"] and payload_by_pci.get(candidate["pci"]) == payload
        if type(candidate.get("label")) is not bool or candidate["label"] != expected:
            _fail(f"candidates[{i}] label does not match CRC and payload")
        normalized_candidates.append({**candidate, "features": normalized_features, "payload": payload,
                                      "offset": _exact_integer(candidate.get('offset'), 'offset'),
                                      "cfo": _finite_number(candidate.get('cfo'), 'cfo')})
    return {**raw, "seed": seed, "users": normalized_users, "candidates": normalized_candidates}


def load_scenes(paths: list[Path]) -> list[dict]:
    """Load JSON scenes and reject records that cannot be independently audited."""
    scenes, seeds = [], set()
    for path in paths:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load scene {path}: {exc}") from exc
        scene = _validate_scene(raw)
        if scene["seed"] in seeds:
            _fail("unique scene IDs (seeds) required")
        seeds.add(scene["seed"])
        scenes.append(scene)
    return scenes


def arrays(scenes: list[dict]) -> dict:
    """Return model-safe measurements plus separately held evaluation truth."""
    rows, labels, groups, pcis, weak = [], [], [], [], []
    true_users, weak_users, scene_ids = {}, {}, []
    for scene in scenes:
        seed, users = scene["seed"], scene["users"]
        scene_ids.append(seed)
        true_users[seed] = len(users)
        min_power = min((u["power_db"] for u in users), default=None)
        weak_pcis = {u["pci"] for u in users if min_power is not None and u["power_db"] == min_power}
        weak_users[seed] = len(weak_pcis)
        for candidate in scene["candidates"]:
            rows.append([candidate["features"][name] for name in FEATURES])
            labels.append(int(candidate["label"]))
            groups.append(seed)
            pcis.append(candidate["pci"])
            weak.append(candidate["pci"] in weak_pcis)
    return {"x": np.asarray(rows, dtype=float).reshape((-1, len(FEATURES))), "y": np.asarray(labels, dtype=int),
            "groups": np.asarray(groups, dtype=int), "pcis": np.asarray(pcis, dtype=int),
            "weak": np.asarray(weak, dtype=bool), "true_users": true_users, "weak_users": weak_users,
            "scene_ids": np.asarray(scene_ids, dtype=int)}
