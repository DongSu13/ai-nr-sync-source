"""Independent whole-campaign audit and scene-paired final statistics."""
from __future__ import annotations
from datetime import datetime
import json
import numpy as np
from scipy.stats import beta
from nr_v4_campaign import DATA,COUNTS,digest,receiver_hash,write_immutable
from nr_v4_audit import require,seq,audit_collection,audit_run
from run_nr_v4_pipeline import check_hashes


def paired_ci(ai,base,reps=5000):
    require([r['seed'] for r in ai]==[r['seed'] for r in base],'unpaired scene IDs')
    den=np.asarray([r['transmitted_users'] for r in ai],int)
    require(np.array_equal(den,[r['transmitted_users'] for r in base]),'unpaired user denominators')
    diff=np.asarray([r['recovered_users']-b['recovered_users'] for r,b in zip(ai,base)],int)
    out=dict(delta=None,ci95=[None,None],scenes=len(ai),repetitions=reps)
    if not den.sum():return out
    out['delta']=float(diff.sum()/den.sum());rng=np.random.default_rng(95341);draws=[]
    for _ in range(reps):
        ix=rng.integers(0,len(ai),len(ai));d=den[ix].sum()
        if d:draws.append(diff[ix].sum()/d)
    out['ci95']=np.quantile(draws,[.025,.975]).tolist() if draws else [None,None]
    return out


def aggregate(rows):
    totals={k:sum(r[k] for r in rows) for k in ('recovered_users','transmitted_users','weak_recovered','weak_transmitted','incorrect_outputs')}
    signal=sum(r['transmitted_users']>0 for r in rows);events=sum(r['incorrect_outputs']>0 for r in rows);n=len(rows)
    totals.update(scenes=n,recovery_rate=totals['recovered_users']/totals['transmitted_users'] if totals['transmitted_users'] else None,
                  weak_recovery_rate=totals['weak_recovered']/totals['weak_transmitted'] if totals['weak_transmitted'] else None,
                  all_recovered_rate=sum(r['all_recovered'] for r in rows)/signal if signal else None,
                  false_output_scenes=events,false_output_upper95=float(beta.ppf(.95,events+1,n-events)) if events<n else 1.)
    for field in ('attempts','searches','seconds','reconstructs','search_seconds','features_seconds','decode_seconds','reconstruct_seconds','score_seconds'):
        if n and field in rows[0]:totals['mean_'+field]=float(np.mean([r[field] for r in rows]))
    return totals


def primary_statement(primary):
    if primary['delta'] is None:return '主终点分母为零，不能判断增益。'
    lo,hi=primary['ci95'];verdict='支持预定义主比较的正增益' if lo is not None and lo>0 else '不支持预定义主比较的正增益'
    if lo is None:return verdict+'，置信区间未定义。'
    return f'{verdict}：差值 {100*primary["delta"]:+.3f} 个百分点，95% 场景配对 CI [{100*lo:+.3f}, {100*hi:+.3f}] 个百分点。'


def audit_all(root=DATA):
    lock=json.loads((root/'freeze.json').read_text())
    check_hashes(lock['source_hashes']);check_hashes(lock['model_hashes']);check_hashes(lock['train_data_hashes']);check_hashes(lock['validation_hashes'])
    require(receiver_hash()==lock['receiver_hash'],'frozen receiver changed')
    frozen_time=datetime.fromisoformat(lock['created_utc']).timestamp();seen=set();splits={}
    for split,count in COUNTS.items():
        records=json.loads((root/'manifests'/f'{split}.json').read_text());ids={r['seed'] for r in records}
        require(len(records)==len(ids)==count and not seen&ids,'manifest overlap or count mismatch');seen|=ids
        expected={r['seed']:r for r in records}
        if split in ('train','development'):
            files=sorted((root/'collections'/split).glob('*.json'));require(len(files)==count,'collection count mismatch')
            for path in files:
                s=json.loads(path.read_text());require(s['seed'] in expected,'unknown collection scene');audit_collection(s,expected[s['seed']])
        elif split=='validation':
            for name in ('classic_selection','rank_selection','gate_selection'):
                summary=json.loads((root/f'{name}_validation_summary.json').read_text());check_hashes(summary['data_hashes'])
        else:
            require(digest(records)==lock['holdout_manifest_hashes'][split],'changed final manifest')
            summary=json.loads((root/f'final_{split}_summary.json').read_text());check_hashes(summary['data_hashes'])
            require(len(summary['data_hashes'])==count,'missing final scenes')
            for file in summary['data_hashes']:
                from pathlib import Path
                path=Path(file);require(path.stat().st_ctime>=frozen_time and path.stat().st_mtime>=frozen_time,'holdout predates freeze')
                s=json.loads(path.read_text());require(s['seed'] in expected,'unknown holdout')
                for row in seq(s['runs']):require(audit_run(row['run'],s['truth'])==row['metrics'],'final recount differs')
        splits[split]=count
    audit=dict(status='PASS',unique_scenes=len(seen),splits=splits,receiver_hash=receiver_hash())
    write_immutable(root/'integrity_audit.json',audit);return audit


def summarize(root=DATA):
    audit=json.loads((root/'integrity_audit.json').read_text());require(audit['status']=='PASS','audit required')
    results={}
    for split in ('test','stress','noise'):
        s=json.loads((root/f'final_{split}_summary.json').read_text());methods={};cells={}
        for name,rows in s['counts'].items():
            budget=name.rsplit('__B',1)[1];base=s['counts'][f'classic__B{budget}']
            methods[name]=dict(**aggregate(rows),exploratory_vs_classic=paired_ci(rows,base))
            for cell in sorted({r['cell'] for r in rows}):
                a=[r for r in rows if r['cell']==cell];b=[r for r in base if r['cell']==cell]
                cells.setdefault(cell,{})[name]=dict(**aggregate(a),exploratory_vs_classic=paired_ci(a,b))
        primary=paired_ci(s['counts']['ai_full_s22__B6'],s['counts']['classic__B6'])
        primary['confirmatory']=split=='test'
        results[split]=dict(primary=primary,methods=methods,cells=cells)
        write_immutable(root/f'{split}_evaluation.json',results[split])
    return results
