import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


def module():
    assert importlib.util.find_spec('nr_v4_results') is not None,'v4 final metrics missing'
    import nr_v4_results
    return nr_v4_results


def test_paired_scene_metric_uses_users_not_mean_of_scene_rates():
    m=module()
    ai=[dict(seed=1,recovered_users=1,transmitted_users=1),dict(seed=2,recovered_users=2,transmitted_users=4)]
    base=[dict(seed=1,recovered_users=0,transmitted_users=1),dict(seed=2,recovered_users=2,transmitted_users=4)]
    r=m.paired_ci(ai,base,reps=200)
    assert r['delta']==pytest.approx(.2)
    base[1]['seed']=3
    with pytest.raises(ValueError,match='paired'):m.paired_ci(ai,base)


def test_noise_recovery_is_undefined_not_zero():
    m=module()
    rows=[dict(seed=1,recovered_users=0,transmitted_users=0,weak_recovered=0,weak_transmitted=0,
               incorrect_outputs=0,all_recovered=False,attempts=6,searches=1,seconds=1.)]
    result=m.aggregate(rows)
    assert result['recovery_rate'] is None and result['weak_recovery_rate'] is None
    assert result['false_output_upper95']==pytest.approx(.95)
    assert m.paired_ci(rows,rows)['ci95']==[None,None]


def test_main_wording_never_claims_gain_for_crossing_ci():
    m=module()
    assert '不支持' in m.primary_statement(dict(delta=.001,ci95=[-.001,.003]))
    assert '支持预定义主比较的正增益' in m.primary_statement(dict(delta=.002,ci95=[.001,.003]))
