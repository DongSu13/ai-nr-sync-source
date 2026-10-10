import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/python'))


def module():
    assert importlib.util.find_spec('run_nr_v3_pipeline') is not None, 'pipeline not implemented'
    import run_nr_v3_pipeline
    return run_nr_v3_pipeline


def test_partition_has_no_overlap_and_covers_remainder():
    ranges=module().partitions(11,3)
    assert ranges==[(1,4),(5,8),(9,11)]
    assert [x for first,last in ranges for x in range(first,last+1)]==list(range(1,12))


def test_small_workload_has_no_empty_workers():
    assert module().partitions(2,3)==[(1,1),(2,2)]


def test_failure_identifies_first_unfinished_scene(tmp_path):
    import json
    records=[dict(seed=1,split='train',cell='mixed',config_hash='a',receiver_hash='r'),
             dict(seed=2,split='train',cell='mixed',config_hash='b',receiver_hash='r')]
    folder=tmp_path/'scenes/train/mixed';folder.mkdir(parents=True)
    (folder/'1.json').write_text(json.dumps({**records[0],'candidates':[{}]*12}))
    assert module().failed_scene_context(records,tmp_path)['seed']==2
