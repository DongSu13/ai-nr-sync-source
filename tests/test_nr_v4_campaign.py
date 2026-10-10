import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


def module():
    assert importlib.util.find_spec('nr_v4_campaign') is not None, 'v4 campaign missing'
    import nr_v4_campaign
    return nr_v4_campaign


def test_scene_splits_and_smoke_cannot_reuse_seed():
    m=module();rows=m.make_manifest();smoke=m.smoke_manifest()
    assert len(rows)==4800 and len(smoke)==32
    assert len({r['seed'] for r in rows+smoke})==4832
    from collections import Counter
    assert Counter(r['split'] for r in rows)==dict(train=1200,validation=400,development=200,test=2000,stress=800,noise=200)
    from nr_v3_campaign import make_manifest as v3
    assert not {r['seed'] for r in rows+smoke}&{r['seed'] for r in v3()}


def test_no_holdout_before_valid_freeze(tmp_path):
    m=module()
    with pytest.raises(ValueError,match='freeze'):m.prepare(tmp_path,'holdout')
    assert not (tmp_path/'manifests/test.json').exists()


def test_smoke_preparation_is_immutable_and_nonzero_noise(tmp_path):
    m=module();rows=m.prepare(tmp_path,'smoke')
    assert rows==m.prepare(tmp_path,'smoke')
    noise=[r for r in rows if r['config']['nu']==0]
    assert len(noise)==4 and all(r['config']['noise_reference']>0 for r in noise)
    import json
    path=tmp_path/'manifests/smoke.json';changed=json.loads(path.read_text())
    changed[0]['seed']+=1;path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='existing'):m.prepare(tmp_path,'smoke')


def test_frozen_holdout_counts_and_noise_scale_verified_before_writes(tmp_path):
    m=module();import json
    rows=[r for r in m.make_manifest() if r['split'] in ('test','stress','noise')]
    for r in rows:
        r['receiver_hash']=m.receiver_hash()
        if r['split']=='noise':r['config']['noise_reference']=.01;r['config_hash']=m.digest(r['config'])
    frozen=dict(receiver_hash=m.receiver_hash(),noise_reference=.01,
                holdout_manifest_hashes={s:m.digest([r for r in rows if r['split']==s]) for s in ('test','stress','noise')})
    (tmp_path/'freeze.json').write_text(json.dumps(frozen))
    assert len(m.prepare(tmp_path,'holdout'))==3000
    frozen['holdout_manifest_hashes']['test']='wrong';(tmp_path/'freeze.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='manifest'):m.prepare(tmp_path,'holdout')
