"""Disk-backed, seed-separated PSS scenes for paper-aligned model ablations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from paper_repro_signal import corr_features, make_scene


def paper_scene(seed: int, snr_db: float | None = None,
                cfo_sc: float | None = None, nid2: int | None = None,
                speed_kmh: float | None = None,
                delay_spread_ns: float | None = None,
                *, curve: bool = False) -> tuple[np.ndarray, dict]:
    if curve and not 1_900_000 <= seed < 1_912_000:
        raise ValueError("curve seed outside locked 12,000-scene range")
    rng = np.random.default_rng(int(seed) + 9_000_001)
    offset = int(rng.integers(8, 91))
    if curve:
        snr_db = float(-5 + (seed - 1_900_000) // 1000)
        cfo_sc, nid2, speed_kmh, delay_spread_ns = 3.5, 2, 0., 100.
    if snr_db is None:
        snr_db = float(rng.integers(-5, 7))
    if cfo_sc is None:
        cfo_sc = float(rng.choice([.8, 2., 3.5]))
    if nid2 is None:
        nid2 = int(rng.integers(0, 3))
    if speed_kmh is None:
        speed_kmh = float(rng.choice([0., 100.]))
    if delay_spread_ns is None:
        delay_spread_ns = float(rng.choice([100., 300.]))
    return make_scene(int(seed), int(nid2), offset, float(cfo_sc),
                      float(snr_db), profile="TDL-A",
                      delay_spread_ns=float(delay_spread_ns),
                      speed_kmh=float(speed_kmh))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_partition(root: Path, name: str, first_seed: int, count: int,
                    *, curve: bool = False) -> dict:
    if count <= 0 or (curve and (first_seed < 1_900_000 or first_seed + count > 1_912_000)):
        raise ValueError("invalid partition count or curve seed range")
    root = Path(root)
    target = root / name
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    shape = (count, 4, 96, 2)
    cross = np.lib.format.open_memmap(target / "x_cross.npy", mode="w+",
                                      dtype=np.float32, shape=shape)
    combined = np.lib.format.open_memmap(target / "x_combined.npy", mode="w+",
                                         dtype=np.float32, shape=shape)
    labels = {key: np.empty(count, dtype)
              for key, dtype in (("seeds", np.int64), ("offset", np.float32),
                                 ("cfo_sc", np.float32), ("nid2", np.int64),
                                 ("snr_db", np.float32),
                                 ("speed_kmh", np.float32),
                                 ("delay_spread_ns", np.float32))}
    for i in range(count):
        seed = first_seed + i
        rx, meta = paper_scene(seed, curve=curve)
        cross[i] = corr_features(rx, combined=False)
        combined[i] = corr_features(rx, combined=True)
        labels["seeds"][i] = seed
        for key in labels:
            if key != "seeds":
                labels[key][i] = meta[key]
        if (i + 1) % 5000 == 0:
            print(f"PAPER DATA {name} {i + 1}/{count}", flush=True)
    cross.flush()
    combined.flush()
    del cross, combined
    np.savez_compressed(target / "labels.npz", **labels)
    manifest = {"name": name, "first_seed": first_seed,
                "last_seed": first_seed + count - 1, "count": count,
                "curve": curve,
                "sha256": {filename: _sha256(target / filename)
                           for filename in ("x_cross.npy", "x_combined.npy", "labels.npz")}}
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_partition(root: Path, name: str) -> dict:
    target = Path(root) / name
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    arrays = {"cross": np.load(target / "x_cross.npy", mmap_mode="r"),
              "combined": np.load(target / "x_combined.npy", mmap_mode="r")}
    with np.load(target / "labels.npz", allow_pickle=False) as labels:
        arrays.update({key: labels[key] for key in labels.files})
    if len(arrays["cross"]) != manifest["count"] or len(arrays["combined"]) != manifest["count"]:
        raise ValueError("partition count disagrees with manifest")
    arrays["manifest"] = manifest
    return arrays
