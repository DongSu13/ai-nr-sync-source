"""Fixed, disk-backed continuous-CFO scenes for the model expansion."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from paper_repro_signal import corr_features, correlation_maps
from pss_cfo_ai import scene_from_seed


WINDOWS = {"fit": (2_100_000, 2_148_000),
           "validation": (2_148_000, 2_160_000),
           "confirm": (2_200_000, 2_203_000)}


def branch_features(rx: np.ndarray, combined: bool) -> np.ndarray:
    cross, half_auto = correlation_maps(rx)
    cross_mag = np.abs(cross)
    cross_scale = max(float(cross_mag.max()), 1e-8)
    channels = [cross_mag / cross_scale, np.angle(cross) / np.pi]
    if combined:
        auto_mag = np.abs(half_auto)
        auto_scale = max(float(auto_mag.max()), 1e-8)
        channels.extend((auto_mag / auto_scale, np.angle(half_auto) / np.pi))
    return np.stack(channels, axis=-1).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_partition(root: Path, name: str, first_seed: int, count: int) -> dict:
    if name not in WINDOWS or count <= 0:
        raise ValueError("unknown or empty partition")
    low, high = WINDOWS[name]
    if first_seed < low or first_seed + count > high:
        raise ValueError("seed range outside locked partition window")
    target = Path(root) / name
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    specs = {"rx": ((count, 384), np.complex64),
             "cross": ((count, 4, 96, 2), np.float32),
             "branch_cross": ((count, 3, 111, 2), np.float32),
             "branch_combined": ((count, 3, 111, 4), np.float32)}
    arrays = {key: np.lib.format.open_memmap(target / f"{key}.npy", mode="w+",
                                             dtype=dtype, shape=shape)
              for key, (shape, dtype) in specs.items()}
    labels = {key: np.empty(count, dtype)
              for key, dtype in (("seeds", np.int64), ("offset", np.int16),
                                 ("nid2", np.int8), ("cfo_sc", np.float32),
                                 ("cfo_center_sc", np.float32),
                                 ("snr_db", np.float32),
                                 ("speed_kmh", np.float32),
                                 ("delay_spread_ns", np.float32))}
    scene_seconds = feature_seconds = 0.
    for i, seed in enumerate(range(first_seed, first_seed + count)):
        began = time.perf_counter()
        rx, truth = scene_from_seed(seed)
        scene_seconds += time.perf_counter() - began
        began = time.perf_counter()
        arrays["rx"][i] = rx
        arrays["cross"][i] = corr_features(rx, combined=False)
        arrays["branch_cross"][i] = branch_features(rx, combined=False)
        arrays["branch_combined"][i] = branch_features(rx, combined=True)
        feature_seconds += time.perf_counter() - began
        labels["seeds"][i] = seed
        for key in labels:
            if key != "seeds":
                labels[key][i] = truth[key]
        if (i + 1) % 5000 == 0:
            print(f"EXPANSION DATA {name} {i + 1}/{count}", flush=True)
    for value in arrays.values():
        value.flush()
    del arrays
    np.savez_compressed(target / "labels.npz", **labels)
    filenames = tuple(f"{key}.npy" for key in specs) + ("labels.npz",)
    manifest = {"name": name, "first_seed": first_seed,
                "last_seed": first_seed + count - 1, "count": count,
                "seconds": {"scene_generation": scene_seconds,
                            "feature_extraction": feature_seconds},
                "sha256": {filename: _sha256(target / filename)
                           for filename in filenames}}
    temporary = target / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(target / "manifest.json")
    return manifest


def load_partition(root: Path, name: str) -> dict:
    target = Path(root) / name
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    result = {key: np.load(target / f"{key}.npy", mmap_mode="r")
              for key in ("rx", "cross", "branch_cross", "branch_combined")}
    with np.load(target / "labels.npz", allow_pickle=False) as source:
        result.update({key: source[key] for key in source.files})
    if any(len(value) != manifest["count"] for value in result.values()):
        raise ValueError("partition count disagrees with manifest")
    result["manifest"] = manifest
    return result
