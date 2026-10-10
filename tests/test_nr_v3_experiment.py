import importlib.util
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/python'))


def module():
    assert importlib.util.find_spec('nr_v3_experiment') is not None, 'v3 experiment not implemented'
    import nr_v3_experiment
    return nr_v3_experiment


def test_selection_uses_seed_mean_not_best_seed_or_test():
    rows=[dict(name='a',family='logistic',seed=s,recovery=r,ap=.8) for s,r in zip([22,23,24],[.9,.3,.3])]
    rows += [dict(name='b',family='logistic',seed=s,recovery=.6,ap=.7) for s in [22,23,24]]
    assert module().select_winners(rows)['logistic']=='b'
    with pytest.raises(ValueError,match='seed'):
        module().select_winners(rows[:-1])


def test_same_scene_cannot_be_both_train_and_validation():
    with pytest.raises(ValueError,match='overlap'):
        module().assert_disjoint([dict(seed=1)],[dict(seed=1)])


def test_scientific_source_change_invalidates_freeze(tmp_path):
    p=tmp_path/'code.py';p.write_text('original')
    hashes=module().hash_files([p])
    module().check_hashes(hashes)
    p.write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        module().check_hashes(hashes)


def test_immutable_result_does_not_overwrite(tmp_path):
    path=tmp_path/'result.json'
    module().write_new_json(path,{'answer':1})
    with pytest.raises(FileExistsError):
        module().write_new_json(path,{'answer':2})
    assert '1' in path.read_text()


def test_invalid_json_payload_never_leaves_a_final_result(tmp_path):
    path=tmp_path/'result.json'
    with pytest.raises(ValueError):module().write_new_json(path,{'value':float('nan')})
    assert not path.exists()


def test_prediction_resume_requires_identical_arrays(tmp_path):
    import numpy as np
    p=tmp_path/'scores.npz'
    module().write_predictions(p,{'y':np.array([0,1])})
    module().write_predictions(p,{'y':np.array([0,1])})
    with pytest.raises(ValueError,match='different'):
        module().write_predictions(p,{'y':np.array([1,1])})


def test_independent_budget_recount_excludes_duplicate_pci_and_missing_users():
    import numpy as np
    d=dict(groups=np.array([1,1,1]),pcis=np.array([10,10,20]),y=np.array([0,1,1]),
           scene_ids=np.array([1]),true_users={1:3})
    assert module().independent_recovery(np.array([.9,.8,.7]),d,2)==(1,3,2)


def test_all_model_families_fit_and_roundtrip_on_real_measurements(tmp_path):
    import joblib
    import numpy as np
    from nr_v3_models import model_specs, make_model
    from threadpoolctl import threadpool_limits
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    rng=np.random.default_rng(8);x=rng.random((512,13));y=(x[:,0]+x[:,1]>1).astype(int)
    for family in ('logistic','hgb','rf','mlp'):
        spec=next(s for s in model_specs() if s['family']==family)
        model=make_model(spec,22)
        with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=2):
            warnings.simplefilter('always');model.fit(x,y)
        assert all(issubclass(w.category,ConvergenceWarning) for w in recorded)
        path=tmp_path/f'{family}.joblib';joblib.dump(model,path)
        with threadpool_limits(limits=2):prediction=joblib.load(path).predict_proba(x)[:,1]
        assert prediction.shape==(512,)
        assert np.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()


def test_smoke_measurements_train_freeze_evaluate_roundtrip(tmp_path,monkeypatch):
    import json
    from nr_v3_data import arrays
    from nr_v3_models import model_specs
    from nr_v3_campaign import write_manifest
    m=module()
    if not (m.DATA/'manifests/smoke.json').exists():
        pytest.skip('requires independently generated MATLAB smoke fixtures')
    smoke,_=m.split_scenes('smoke')
    examples=[s for s in smoke if any(c['label'] for c in s['candidates'])]
    assert len(examples)>=32
    cursor=0
    for split,size in [('train',12),('validation',8),('development',4),('test',8)]:
        monkeypatch.setitem(m.COUNTS,split,size)
        records=[]
        for scene in examples[cursor:cursor+size]:
            copied={**scene,'split':split,'suite':split,'cell':'unit'}
            m.write_new_json(tmp_path/'scenes'/split/'unit'/f"{copied['seed']}.json",copied)
            records.append({k:copied[k] for k in ('seed','split','suite','cell','config','config_hash','receiver_hash')})
        write_manifest(tmp_path/'manifests'/f'{split}.json',records);cursor+=size
    ts,th=m.split_scenes('train',tmp_path);vs,vh=m.split_scenes('validation',tmp_path)
    t,v=arrays(ts),arrays(vs)
    recipe=dict(train_hash=th,validation_hash=vh,source_hashes=m.scientific_hashes())
    spec=model_specs()[0]
    rows=[m.fit_one(spec,seed,t,v,recipe,root=tmp_path) for seed in m.SEEDS]
    ablations=[]
    for variant,removed in m.ABLATIONS.items():
        for seed in m.SEEDS:
            ablations.append(m.fit_one(spec,seed,t,v,recipe,variant,tuple(f for f in m.FEATURES if f not in removed),tmp_path))
    m.write_new_json(tmp_path/'training_summary.json',dict(recipe=recipe,rows=rows,ablations=ablations,
                     family_winners={'logistic':spec['name']},selected_ai=spec['name']))
    lock=m.freeze(tmp_path)
    assert lock['selected_ai'].endswith('__s22')
    result=m.evaluate_suite('test',tmp_path)
    assert result['scenes']==8 and result['candidates']==96
    assert result['methods'][lock['selected_ai']]['budgets']['2']['transmitted_users']>=8
    for name,values in result['methods'].items():
        assert len(values['exploratory_vs_classical']['ci95'])==2
        assert values['exploratory_vs_classical']['scenes']==8
        assert 'exploratory_vs_classical' in result['cells']['unit'][name]
    assert result['methods'][lock['selected_classical']]['exploratory_vs_classical']['ci95']==[0.,0.]
    with pytest.raises(FileExistsError):m.freeze(tmp_path)
    with pytest.raises(FileExistsError):m.evaluate_suite('test',tmp_path)
