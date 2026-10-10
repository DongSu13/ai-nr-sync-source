import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))


def campaign():
    assert importlib.util.find_spec('nr_v3_campaign') is not None, 'v3 manifest generator is not implemented'
    import nr_v3_campaign
    return nr_v3_campaign


def test_manifest_disjoint_suites_and_exact_workload():
    m = campaign().make_manifest()
    assert len(m) == 11100
    assert len({r['seed'] for r in m}) == 11100
    assert Counter(r['split'] for r in m) == dict(train=3000, validation=1000, development=500,
                                              test=2000, sweep=3200, stress=800, noise=600)
    assert len({r['cell'] for r in m if r['split'] == 'sweep'}) == 32
    for r in m:
        assert r['config_hash'] == campaign().config_hash(r['config'])
        if r['split'] in ('train', 'validation', 'development', 'test'):
            assert r['config']['nu'] in (1, 2, 3, 4)
            assert r['config']['profile'] != 'TDL-D'


def test_smoke_does_not_reuse_formal_seeds_and_covers_boundaries():
    module = campaign()
    smoke = module.smoke_manifest()
    assert len(smoke) == 64
    assert not {r['seed'] for r in smoke} & {r['seed'] for r in module.make_manifest()}
    assert any(r['config']['nu'] == 0 for r in smoke)
    assert any(r['config']['nu'] == 6 for r in smoke)
    assert any(r['config']['profile'] == 'TDL-D' for r in smoke)
    assert any(r['config']['cfo_max'] == 1500 for r in smoke)


def test_manifest_existing_conflict_is_not_overwritten(tmp_path):
    module = campaign()
    record = module.smoke_manifest()[0]
    path = tmp_path / 'manifest.json'
    module.write_manifest(path, [record])
    module.write_manifest(path, [record])
    with pytest.raises(ValueError, match='different'):
        module.write_manifest(path, [{**record, 'seed': record['seed'] + 1}])


def test_holdout_generation_requires_freeze_and_noise_reference(tmp_path):
    module = campaign()
    with pytest.raises(ValueError, match='freeze'):
        module.prepare(tmp_path, 'holdout')
