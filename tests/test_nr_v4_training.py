import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


def mod():
    assert importlib.util.find_spec('nr_v4_train') is not None,'v4 training orchestration missing'
    import nr_v4_train
    return nr_v4_train


def test_validation_selection_uses_all_seeds_not_best_seed():
    m=mod()
    rows=[dict(config='a',seed=s,recovered=n,transmitted=10,incorrect=0,attempts=6,complexity=1)
          for s,n in [(22,9),(23,1),(24,1)]]
    rows += [dict(config='b',seed=s,recovered=5,transmitted=10,incorrect=0,attempts=6,complexity=1) for s in (22,23,24)]
    assert m.select_config(rows)=='b'
    with pytest.raises(ValueError,match='seed'):m.select_config(rows[:-1])


def test_job_manifest_identity_and_singleton_policy(tmp_path):
    m=mod();policies=[dict(name='classic',policy=dict(rank='dmrs_fit',gate='always',mode='sic'))]
    first=m.write_job('unit',policies,[6],tmp_path)
    assert first==m.write_job('unit',policies,[6],tmp_path)
    with pytest.raises(ValueError,match='existing'):m.write_job('unit',policies,[2],tmp_path)


def test_train_arrays_refuse_any_nontraining_split():
    m=mod()
    with pytest.raises(ValueError,match='train'):m.training_arrays([dict(split='test')])
