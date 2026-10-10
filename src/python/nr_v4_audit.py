"""Independent trace accounting. This code never calls the online controller."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from nr_v4_campaign import DATA,receiver_hash


def seq(value):
    if isinstance(value,dict):return [value]
    if not isinstance(value,list):raise ValueError('expected MATLAB JSON record array')
    return value


def require(condition,message):
    if not condition:raise ValueError(message)


def payload(value):
    require(isinstance(value,list) and len(value)==24 and all(type(v) is int and v in (0,1) for v in value),'invalid payload')
    return value


def counts(outputs,truth):
    users=seq(truth['users']);out=seq(outputs)
    require(len({u['pci'] for u in users})==len(users),'duplicate truth PCI')
    require(len({o['pci'] for o in out})==len(out),'duplicate output PCI')
    by_pci={u['pci']:payload(u['payload']) for u in users}
    good={o['pci'] for o in out if by_pci.get(o['pci'])==payload(o['payload'])}
    minimum=min((u['power_db'] for u in users),default=0)
    weak={u['pci'] for u in users if u['power_db']==minimum}
    return dict(recovered_users=len(good),transmitted_users=len(users),weak_recovered=len(good & weak),
                weak_transmitted=len(weak),incorrect_outputs=len(out)-len(good),
                all_recovered=bool(users) and len(good)==len(users))


def audit_run(run,truth,start_decode=0,start_search=None,initial_outputs=(),initial_attempted=()):
    events=seq(run['events']);outputs=seq(run['outputs']);used=start_decode
    round_=(int(run['budget']>0) if start_search is None else start_search)
    expected=[dict(o) for o in initial_outputs];seen={o['pci'] for o in expected};attempted=set(initial_attempted);reconstructs=0
    require(0<=used<=run['decode_count']<=run['budget'],'invalid decode budget')
    require(0<=round_<=run['search_count']<=run['max_searches'],'invalid search budget')
    for e in events:
        f=np.asarray(e['features'],float)
        require(f.shape==(17,) and np.isfinite(f).all(),'invalid observation features')
        require(e['attempt']==used+1 and e['residual_version']==round_,'attempt/round sequence mismatch')
        require(list(f[13:15])==[round_,used] and f[16]==len(expected),'feature state mismatch')
        require(e['pci'] not in seen and e['pci'] not in attempted,'repeated output or same-state PCI attempt')
        attempted.add(e['pci']);used+=1;payload(e['payload'])
        require(type(e['crc']) is bool and type(e['cancelled']) is bool,'CRC/cancel must be boolean')
        if e['crc']:
            expected.append(dict(pci=e['pci'],payload=e['payload'],attempt=used,round=round_));seen.add(e['pci'])
        g=np.asarray(e['gate_features'],float)
        if g.size:
            reconstructs+=1
            require(g.shape==(21,) and np.isfinite(g).all() and np.array_equal(g[:17],f),'bad gate observation')
            require(g[-1]==run['budget']-used and used<run['budget'] and round_<run['max_searches'],'gate budget leak')
            require(e['crc'],'reconstruction before CRC')
        if e['cancelled']:
            require(e['crc'] and g.size==21,'cancel without confirmed reconstruction')
            round_+=1;attempted=set()
        else:require(e['energy_after']==e['energy_before'],'rejected cancellation changed residual')
    require(used==run['decode_count'] and len(events)==used-start_decode,'unlogged decode')
    require(round_==run['search_count'],'unlogged search')
    require(reconstructs==run['reconstruct_count'],'unlogged reconstruction')
    require(outputs==expected,'outputs differ from CRC events')
    return counts(outputs,truth)


def audit_collection(sample,record,current_receiver=True):
    for key in ('seed','split','cell','schema','config','config_hash','receiver_hash'):
        require(sample[key]==record[key],f'manifest mismatch {key}')
    if current_receiver:require(sample['receiver_hash']==receiver_hash(),'receiver changed')
    truth=sample['truth'];users=seq(truth['users']);require(len(users)==record['config']['nu'],'wrong user count')
    require(len({u['channel_seed'] for u in users})==len(users),'reused user channel seed')
    online=audit_run(sample['online_run'],truth)
    rows=seq(sample['rank_records']);x=np.asarray(sample['rank_x'],float).reshape((-1,17));y=np.asarray(sample['rank_y']).reshape(-1)
    require(len(x)==len(y)==len(rows) and np.isfinite(x).all(),'rank row count/feature mismatch')
    by_pci={u['pci']:u['payload'] for u in users}
    for i,row in enumerate(rows):
        label=bool(row['crc'] and by_pci.get(row['pci'])==payload(row['payload']))
        require(row['label']==label and y[i]==label,'rank truth label mismatch')
        require(x[i,13]==row['round'] and x[i,14]==row['used'],'rank state mismatch')
    branches=seq(sample['branches']);require(len(branches)<=4,'too many branches')
    for b in branches:
        initial=[o for o in seq(sample['online_run']['outputs']) if o['attempt']<=b['start_decode_count']]
        old=counts(initial,truth)['recovered_users']
        evaluated=[]
        for method,offset in (('skip',0),('cancel',1)):
            run=b[method];start=b['start_decode_count']
            require(run['budget']==min(6,start+2) and run['max_searches']==4,'branch budget differs from teacher')
            tried=() if offset else [e['pci'] for e in seq(sample['online_run']['events'])
                                    if e['attempt']<=start and e['residual_version']==b['start_search_count']]
            m=audit_run(run,truth,start,b['start_search_count']+offset,initial,tried)
            require(run['decode_count']-start==b[f'{method}_decode_calls']<=2,'branch hidden decode')
            require(m['recovered_users']-old==b[f'{method}_new_users'],'branch user label mismatch')
            evaluated.append(m['recovered_users'])
        require(b['benefit']==evaluated[1]-evaluated[0],'branch benefit mismatch')
        require(np.asarray(b['features']).shape==(21,) and np.isfinite(b['features']).all(),'invalid gate row')
    return dict(seed=int(sample['seed']),rank_rows=len(rows),positives=int(y.sum()),branches=len(branches),online=online)


def load_collection(split,root=DATA,complete=True):
    root=Path(root);records=json.loads((root/'manifests'/f'{split}.json').read_text())
    paths=sorted((root/'collections'/split).glob('*.json'));expected={r['seed']:r for r in records}
    if complete:require(len(paths)==len(records),'incomplete collection')
    samples=[];seen=set()
    for path in paths:
        s=json.loads(path.read_text());seed=s['seed']
        require(seed in expected and seed not in seen,'unknown/duplicate scene');seen.add(seed)
        audit_collection(s,expected[seed]);samples.append(s)
    return samples
