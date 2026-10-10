import copy
import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


def fixture():
    payload=[0]*24
    truth={'users':[{'pci':2,'payload':payload,'power_db':0},{'pci':3,'payload':[1]*24,'power_db':-6}]}
    def event(attempt,round_,pci,crc,cancelled):
        f=[.1]*13+[round_,attempt-1,1,0 if attempt<3 else 1]
        return dict(attempt=attempt,residual_version=round_,candidate_id=pci,pci=pci,crc=crc,
                    payload=payload,features=f,cancelled=cancelled,energy_before=1,energy_after=.5 if cancelled else 1,
                    gate_features=f+[1,.5,.1,3-attempt] if cancelled else [])
    run=dict(budget=3,max_searches=4,decode_count=3,search_count=2,reconstruct_count=1,
             outputs=[dict(pci=2,payload=payload,attempt=2,round=1)],
             events=[event(1,1,1,False,False),event(2,1,2,True,True),event(3,2,1,False,False)])
    return run,truth


def audit():
    assert importlib.util.find_spec('nr_v4_audit') is not None,'v4 audit missing'
    from nr_v4_audit import audit_run
    return audit_run


def test_online_recovery_counted_independently_with_weak_denominator():
    fn=audit();r,t=fixture();m=fn(r,t)
    assert m['recovered_users']==1 and m['transmitted_users']==2
    assert m['weak_recovered']==0 and m['weak_transmitted']==1


@pytest.mark.parametrize('corruption',['budget','unlogged_decode','crc','duplicate','round','feature_state'])
def test_corrupted_online_trace_is_rejected(corruption):
    fn=audit();r,t=fixture()
    if corruption=='budget':r['budget']=2
    if corruption=='unlogged_decode':r['decode_count']=4
    if corruption=='crc':r['events'][1]['crc']=False
    if corruption=='duplicate':r['outputs']*=2
    if corruption=='round':r['events'][2]['residual_version']=1
    if corruption=='feature_state':r['events'][2]['features'][14]=1
    with pytest.raises(ValueError):fn(r,t)


def test_actual_collection_labels_and_branches_recount():
    audit()
    from nr_v4_audit import load_collection,audit_collection,seq
    from nr_v4_campaign import DATA
    import json
    if not (DATA/'collections/train/40000001.json').exists():pytest.skip('real engineering integration fixture absent')
    sample=json.loads((DATA/'collections/train/40000001.json').read_text())
    record=json.loads((DATA/'manifests/train.json').read_text())[0]
    assert audit_collection(sample,record)['rank_rows']>0
    bad=copy.deepcopy(sample);bad['rank_y'][0]=1-bad['rank_y'][0]
    with pytest.raises(ValueError,match='label'):audit_collection(bad,record)
    bad=copy.deepcopy(sample);seq(bad['branches'])[0]['benefit']+=1
    with pytest.raises(ValueError,match='benefit'):audit_collection(bad,record)


def test_continuation_cannot_retry_already_attempted_same_residual():
    fn=audit();r,t=fixture();initial=r['outputs']
    r['events']=r['events'][2:];r['events'][0]['residual_version']=1;r['events'][0]['features'][13]=1
    r['search_count']=1;r['reconstruct_count']=0
    with pytest.raises(ValueError,match='same-state'):
        fn(r,t,start_decode=2,start_search=1,initial_outputs=initial,initial_attempted=(1,2))
