import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


def test_exclusive_campaign_lock_blocks_second_job(tmp_path):
    assert importlib.util.find_spec('run_nr_v4_pipeline') is not None,'v4 supervisor missing'
    from run_nr_v4_pipeline import campaign_lock
    with campaign_lock(tmp_path):
        assert (tmp_path/'pipeline.lock').exists()
        with pytest.raises(FileExistsError):
            with campaign_lock(tmp_path):pytest.fail('second owner acquired lock')
    assert not (tmp_path/'pipeline.lock').exists()


def test_source_snapshot_refuses_changed_file(tmp_path):
    assert importlib.util.find_spec('run_nr_v4_pipeline') is not None,'v4 supervisor missing'
    from run_nr_v4_pipeline import hash_files,check_hashes
    p=tmp_path/'receiver.m';p.write_text('first')
    snapshot=hash_files([p]);check_hashes(snapshot)
    p.write_text('changed')
    with pytest.raises(ValueError,match='changed'):check_hashes(snapshot)
