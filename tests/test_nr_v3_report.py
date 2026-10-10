import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


def test_report_does_not_turn_negative_primary_into_gain():
    assert importlib.util.find_spec('report_nr_v3') is not None, 'reporter not implemented'
    from report_nr_v3 import primary_statement
    result={'delta_recovery':-.01,'ci95':[-.02,.003]}
    assert '不支持' in primary_statement(result)
    assert '-1.00' in primary_statement(result)
    assert '支持主终点正增益' in primary_statement({'delta_recovery':.02,'ci95':[.001,.04]})


def test_report_requires_verified_counts():
    assert importlib.util.find_spec('report_nr_v3') is not None, 'reporter not implemented'
    from report_nr_v3 import render_report
    with pytest.raises(ValueError,match='audit'):
        render_report({}, {}, {'status':'FAIL'}, {})


def test_detail_tables_preserve_negative_ci_and_measured_timing():
    from report_nr_v3 import detail_tables
    result={'methods':{'ai':{'budgets':{'2':{'recovery_rate':.5}},'exploratory_vs_classical':{'delta_recovery':-.01,'ci95':[-.02,.01]}}},
            'score_seconds':{'ai':.25},'candidates':1000,'median_scene_timing':{'features_seconds':.1},
            'cells':{}}
    lock={'selected_ai':'ai','selected_classical':'base','family_winners':{}}
    text=detail_tables({'test':result,'sweep':{'cells':{}}},lock)
    assert '-1.00' in text and '-2.00' in text
    assert '0.250' in text
    assert '探索性' in text


def test_budget_table_separates_pool_ceiling_and_actual_recovery():
    from report_nr_v3 import budget_table
    budgets={str(b):{'recovered_users':b,'transmitted_users':10,'recovery_rate':b/10,
                     'available_coverage':.7,'recovery_of_available':b/7,'decode_attempts':2*b}
             for b in (1,2,4,6)}
    text=budget_table({'methods':{'ai':{'budgets':budgets}}},['ai'])
    assert '70.00%' in text and '28.57%' in text
    assert '2 / 10' in text
    assert '候选池' in text
