import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


def test_report_requires_complete_audited_campaign():
    assert importlib.util.find_spec('report_nr_v4') is not None,'v4 reporter missing'
    from report_nr_v4 import render_report
    with pytest.raises(ValueError,match='audit'):render_report({}, {}, {'status':'FAIL'}, {})


def test_complete_report_preserves_negative_primary_and_actual_timings():
    from report_nr_v4 import render_report
    from nr_v4_results import aggregate
    row=dict(seed=1,recovered_users=1,transmitted_users=2,weak_recovered=0,weak_transmitted=1,
             incorrect_outputs=0,all_recovered=False,attempts=6,searches=2,seconds=.5,reconstructs=1,
             search_seconds=.1,features_seconds=.1,decode_seconds=.1,reconstruct_seconds=.1,score_seconds=.1)
    metric=aggregate([row]);metric['exploratory_vs_classic']=dict(delta=-.01,ci95=[-.02,.01])
    methods={'ai_full_s22__B6':metric,'classic__B6':metric}
    results={'test':dict(primary=dict(delta=-.01,ci95=[-.02,.01]),methods=methods),
             'stress':dict(cells={'stress_1':methods}),'noise':dict(methods=methods)}
    lock=dict(selected_classical={'policy':{'rank':'dmrs_fit'}},selected_rank='logistic_0.1',selected_gate={'config':'ridge_1'})
    text=render_report(results,lock,{'status':'PASS','unique_scenes':4800},{'models':{'unit':{'warnings':[]}}})
    assert '不支持预定义主比较' in text and '-1.000' in text and '0.5000' in text
