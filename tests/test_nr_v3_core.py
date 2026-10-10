import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))


def _candidate(pci, *, crc=False, payload=None, **features):
    base = {name: 1.0 for name in (
        "pss", "sss", "correlation", "peak_ratio", "consistency", "abs_cfo",
        "dmrs_power", "dmrs_coherence", "dmrs_fit_error", "dmrs_variation",
        "llr_abs", "llr_low_fraction", "estimated_noise",
    )}
    base.update(features)
    return {"pci": pci, "offset": 0, "cfo": 0.0, "features": base,
            "crc": crc, "payload": list(payload or [0] * 24), "label": crc}


def _scene(seed=1, users=None, candidates=None):
    users = users if users is not None else [
        {"pci": 10, "payload": [1] * 24, "power_db": 0.0, "channel_seed": 1},
        {"pci": 20, "payload": [0, 1] * 12, "power_db": -6.0, "channel_seed": 2},
        {"pci": 30, "payload": [1, 0] * 12, "power_db": -6.0, "channel_seed": 3},
    ]
    candidates = candidates if candidates is not None else [
        _candidate(10, correlation=9.0),
        _candidate(10, crc=True, payload=[1] * 24, correlation=8.0),
        _candidate(20, crc=True, payload=[0, 1] * 12, correlation=7.0),
    ] + [_candidate(100 + i, correlation=float(i)) for i in range(9)]
    return {"seed": seed, "split": "test", "suite": "id", "cell": "id",
            "config": {}, "config_hash": "h", "schema": "nr_gain_v3", "users": users,
            "candidates": candidates, "receiver_hash": "r", "timings": {},
            "ref_power": 1.0, "noise_variance": 1.0}


def test_load_and_budget_deduplicate_pci_and_count_unavailable_users(tmp_path):
    from nr_v3_data import arrays, load_scenes
    from nr_v3_metrics import budget_counts, summarize_counts
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_scene()), encoding="utf-8")
    data = arrays(load_scenes([path]))
    counts = budget_counts(np.arange(12, 0, -1), data, 2)
    assert counts.tolist() == [[1, 3, 1, 2, 0, 2, 2]]
    summary = summarize_counts(counts)
    assert summary["recovery_rate"] == pytest.approx(1 / 3)
    assert summary["available_coverage"] == pytest.approx(2 / 3)
    assert summary['recovery_of_available'] == .5


def test_noise_and_invalid_records_are_handled_without_repair(tmp_path):
    from nr_v3_data import arrays, load_scenes
    from nr_v3_metrics import budget_counts, summarize_counts
    noise = _scene(2, users=[], candidates=[_candidate(100 + i) for i in range(12)])
    path = tmp_path / "noise.json"
    path.write_text(json.dumps(noise), encoding="utf-8")
    d = arrays(load_scenes([path]))
    assert d["scene_ids"].tolist() == [2]
    assert summarize_counts(budget_counts(np.ones(12), d, 2))["recovery_rate"] is None
    bad = _scene(3)
    bad["candidates"][0]["features"]["crc"] = 1
    invalid = tmp_path / "bad.json"
    invalid.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="features"):
        load_scenes([invalid])
    bad = _scene(4)
    bad["candidates"][0]["features"]["pss"] = float("nan")
    invalid.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_scenes([invalid])
    bad = _scene(5)
    bad["candidates"][1]["label"] = False
    invalid.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        load_scenes([invalid])


def test_metrics_threshold_candidate_stats_and_paired_ci():
    from nr_v3_metrics import candidate_metrics, paired_user_ci, threshold_at_fpr
    scores, y = np.array([.9, .9, .5, .1]), np.array([1, 0, 1, 0])
    assert threshold_at_fpr(scores, y, .0) > .9
    m = candidate_metrics(scores, y, .5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 0)
    ai = np.array([[2, 2, 0, 0, 1, 2, 2], [0, 0, 0, 0, 0, 2, 0]])
    base = np.array([[1, 2, 0, 0, 0, 2, 2], [0, 0, 0, 0, 0, 2, 0]])
    ci = paired_user_ci(ai, base, reps=50, seed=7)
    assert ci["delta_recovery"] == pytest.approx(.5)
    assert ci["scenes"] == 2 and ci["repetitions"] == 50
    assert ci['ci95'] == [.5, .5]
    assert ci['undefined_replicates'] > 0


def test_models_keep_original_input_shape_for_ablation():
    from nr_v3_data import FEATURES
    from nr_v3_models import CLASSICAL, classical_scores, make_model, model_specs
    x = np.vstack([np.zeros(13), np.ones(13), np.full(13, .1), np.full(13, .9)] * 8)
    y = np.array([0, 1, 0, 1] * 8)
    assert set(CLASSICAL) == {"correlation", "sss", "dmrs_coherence", "llr_abs", "dmrs_fit", "joint_pilot"}
    assert classical_scores(x, "dmrs_fit").shape == (len(x),)
    specs = model_specs()
    assert {s["family"] for s in specs} == {"logistic", "hgb", "rf", "mlp"}
    assert len(specs) == 24
    model = make_model(next(s for s in specs if s["family"] == "logistic"), 22,
                       tuple(f for f in FEATURES if f != "pss"))
    model.fit(x, y)
    assert model.predict_proba(x).shape == (len(x), 2)
    with pytest.raises(ValueError):
        make_model(specs[0], 22, ("bad",))


def test_zero_budget_and_frozen_model_roundtrip(tmp_path):
    from nr_v3_data import arrays, load_scenes
    from nr_v3_metrics import budget_counts
    from nr_v3_models import make_model, model_specs
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(_scene()), encoding="utf-8")
    d = arrays(load_scenes([scene_path]))
    assert budget_counts(np.ones(12), d, 0)[0, [0, 5]].tolist() == [0, 0]
    x = np.vstack([np.zeros(13), np.ones(13)] * 8)
    model = make_model(next(s for s in model_specs() if s["family"] == "logistic"), 22)
    model.fit(x, np.array([0, 1] * 8))
    frozen = tmp_path / "frozen.joblib"
    joblib.dump(model, frozen)
    assert joblib.load(frozen).predict_proba(x).shape == (len(x), 2)


def test_threshold_handles_large_tied_score_arrays():
    from nr_v3_metrics import threshold_at_fpr
    scores = np.repeat(np.arange(1000, dtype=float), 12)
    y = np.tile(np.array([1] * 6 + [0] * 6), 1000)
    threshold = threshold_at_fpr(scores, y, .1)
    selected = scores >= threshold
    assert (selected & (y == 0)).sum() / (y == 0).sum() <= .1


def test_matlab_scientific_notation_seed_is_exact_integer(tmp_path):
    from nr_v3_data import load_scenes
    record = _scene(19000027.0)
    path = tmp_path / 'matlab.json'
    path.write_text(json.dumps(record))
    scene = load_scenes([path])[0]
    assert scene['seed'] == 19000027 and type(scene['seed']) is int
    for invalid in (1.5, True, float('inf'), 2**53 + 1):
        record['seed'] = invalid
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            load_scenes([path])


@pytest.mark.parametrize('field,value', [('suite',None),('cell',4),('config_hash',[]),('receiver_hash',''),('timings',{'seconds':-1})])
def test_loader_rejects_malformed_audit_metadata(tmp_path,field,value):
    from nr_v3_data import load_scenes
    record=_scene();record[field]=value;path=tmp_path/'bad.json';path.write_text(json.dumps(record))
    with pytest.raises(ValueError):load_scenes([path])


def test_loader_rejects_invalid_cfo_offset_and_channel_seed(tmp_path):
    from nr_v3_data import load_scenes
    path=tmp_path/'bad.json'
    for object_name,field,value in [('candidates','cfo',float('nan')),('candidates','offset',1.5),('users','channel_seed',True)]:
        record=_scene();record[object_name][0][field]=value;path.write_text(json.dumps(record))
        with pytest.raises(ValueError):load_scenes([path])
