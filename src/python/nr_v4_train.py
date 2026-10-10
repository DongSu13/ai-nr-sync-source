"""Train-only model fits and actual closed-loop, validation-only selection."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import sklearn
import scipy
from threadpoolctl import threadpool_limits
from nr_v4_campaign import ROOT,DATA,digest,receiver_hash,write_immutable,make_manifest
from nr_v4_models import specs,make_model,export_model,predict_export
from nr_v4_audit import seq,require,audit_run,load_collection
from run_nr_v4_pipeline import hash_files,check_hashes,collection_sources,matlab_job,state

SEEDS=(22,23,24)
CLASSIC=('correlation','sss','dmrs_coherence','llr_abs','dmrs_fit','joint_pilot')


def scientific_files():
    return collection_sources()+sorted((ROOT/'src/python').glob('nr_v4_*.py'))+[ROOT/'src/python/run_nr_v4_pipeline.py']


def safe(name):return name.replace('.','p').replace('-','m')


def select_config(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['config']].append(row)
    ranked=[]
    for name,group in groups.items():
        require(sorted(r['seed'] for r in group)==list(SEEDS),'missing or duplicate training seed')
        require(all(r['transmitted']>0 for r in group),'empty validation denominator')
        key=(-np.mean([r['recovered']/r['transmitted'] for r in group]),
             np.mean([r['incorrect'] for r in group]),np.mean([r['attempts'] for r in group]),
             np.mean([r['complexity'] for r in group]),name)
        ranked.append((key,name))
    require(bool(ranked),'no configurations');return min(ranked)[1]


def write_job(name,policies,budgets,root=DATA):
    job=dict(name=name,policies=policies,budgets=budgets,receiver_hash=receiver_hash())
    job['protocol_id']=digest(job);path=Path(root)/'jobs'/f'{name}.json'
    write_immutable(path,job);return path


def training_arrays(samples):
    require(bool(samples) and all(s['split']=='train' for s in samples),'training arrays must use train only')
    x=np.concatenate([np.asarray(s['rank_x'],float).reshape(-1,17) for s in samples])
    y=np.concatenate([np.asarray(s['rank_y'],int).reshape(-1) for s in samples])
    branches=[b for s in samples for b in seq(s['branches'])]
    require(bool(branches),'no CRC branch training states; protocol review required')
    gx=np.asarray([b['features'] for b in branches],float);gy=np.asarray([b['benefit'] for b in branches],float)
    require(x.shape[1]==17 and gx.shape[1]==21 and np.isfinite(x).all() and np.isfinite(gx).all(),'bad feature matrix')
    require(set(np.unique(y))=={0,1},'rank labels need both classes')
    return dict(rank=(x,y),gate=(gx,gy))


def fit_models():
    require(not (DATA/'freeze.json').exists(),'models already frozen')
    samples=load_collection('train');arrays=training_arrays(samples)
    recipe=dict(source_hashes=hash_files(scientific_files()),data_hashes=hash_files(sorted((DATA/'collections/train').glob('*.json'))),
                numpy_version=np.__version__,sklearn_version=sklearn.__version__,scipy_version=scipy.__version__)
    folder=DATA/'models';folder.mkdir(exist_ok=True);models={}
    for task in ('rank','gate'):
        x,y=arrays[task]
        for spec in specs(task):
            for seed in SEEDS:
                name=f'{task}_{safe(spec["name"])}_s{seed}';meta=folder/f'{name}.json';binary=folder/f'{name}.joblib'
                recipe_id=digest(dict(recipe=recipe,spec=spec,seed=seed))
                if meta.exists():
                    row=json.loads(meta.read_text());require(row['recipe_id']==recipe_id,'fit cache recipe changed');check_hashes(row['model_hash'])
                else:
                    require(not binary.exists(),'orphan model file requires review')
                    model=make_model(spec,seed);t=time.perf_counter()
                    with threadpool_limits(1),warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter('always');model.fit(x,y)
                        exported=export_model(model,task);probe=x[::max(1,len(x)//100)]
                        expected=model.predict_proba(probe)[:,1] if task=='rank' else model.predict(probe)
                        np.testing.assert_allclose(predict_export(exported,probe),expected,atol=1e-10,rtol=1e-10)
                    tmp=binary.with_suffix('.joblib.tmp');joblib.dump(model,tmp);tmp.rename(binary)
                    row=dict(name=name,task=task,spec=spec,seed=seed,recipe_id=recipe_id,artifact=exported,
                             training_seconds=time.perf_counter()-t,warnings=[str(w.message) for w in recorded],model_hash=hash_files([binary]),
                             complexity=sum(len(tree) for tree in exported.get('trees',[])) or len(exported.get('coef',[])))
                    write_immutable(meta,row)
                models[name]=row;print('FIT',name,'seconds',row['training_seconds'],flush=True)
    write_immutable(DATA/'training_summary.json',dict(recipe=recipe,models=models))
    return models


def evaluate_job(path,split='validation'):
    job=json.loads(Path(path).read_text());records=json.loads((DATA/'manifests'/f'{split}.json').read_text())
    require(job['receiver_hash']==receiver_hash(),'job receiver changed')
    snapshot=hash_files(scientific_files())
    matlab_job(f"run_nr_v4('{split}',1,{len(records)},'evaluate','{Path(path).as_posix()}')",f'{job["name"]}_{split}')
    check_hashes(snapshot)
    folder=DATA/'campaigns'/job['name']/split;files=sorted(folder.glob('*.json'))
    require(len(files)==len(records),'incomplete policy evaluation')
    expected_records={r['seed']:r for r in records};totals={};scene_counts={};seen=set()
    expected_keys={(p['name'],b) for p in job['policies'] for b in p.get('budgets',job['budgets'])}
    for file in files:
        result=json.loads(file.read_text());seed=result['seed'];require(seed not in seen and seed in expected_records,'scene overlap');seen.add(seed)
        record=expected_records[seed]
        for key in ('seed','split','cell','config','config_hash','receiver_hash'):require(result[key]==record[key],f'policy scene mismatch {key}')
        require(result['protocol_id']==job['protocol_id'],'policy changed')
        rows=seq(result['runs']);keys=[(r['name'],r['budget']) for r in rows]
        require(set(keys)==expected_keys and len(keys)==len(expected_keys),'missing/duplicate policy run')
        for row in rows:
            run=row['run'];require(run['budget']==row['budget'] and run['max_searches']==4,'unequal runtime budgets')
            m=audit_run(run,result['truth']);require(m==row['metrics'],'MATLAB metric recount mismatch')
            key=f'{row["name"]}__B{row["budget"]}'
            if key not in totals:totals[key]=dict(recovered=0,transmitted=0,incorrect=0,attempts=0,searches=0)
            for dest,source in (('recovered','recovered_users'),('transmitted','transmitted_users'),('incorrect','incorrect_outputs')):totals[key][dest]+=m[source]
            totals[key]['attempts']+=run['decode_count'];totals[key]['searches']+=run['search_count']
            scene_counts.setdefault(key,[]).append(dict(seed=int(seed),cell=result['cell'],**m,
                attempts=run['decode_count'],searches=run['search_count'],seconds=run['timings']['total_seconds'],
                reconstructs=run['reconstruct_count'],**{k:v for k,v in run['timings'].items() if k!='total_seconds'}))
    summary=dict(job_id=job['protocol_id'],split=split,totals=totals,counts=scene_counts,data_hashes=hash_files(files))
    write_immutable(DATA/f'{job["name"]}_{split}_summary.json',summary)
    return summary


def select_and_freeze():
    if (DATA/'freeze.json').exists():
        lock=json.loads((DATA/'freeze.json').read_text());check_hashes(lock['source_hashes']);check_hashes(lock['model_hashes']);return lock
    models=fit_models();state('validating_classic')
    classic=[dict(name=f'classic_{rank}_{safe(gate)}',policy=dict(rank=rank,gate=gate,mode='sic'))
             for rank in CLASSIC for gate in ('always','energy_0.95','energy_1')]
    result=evaluate_job(write_job('classic_selection',classic,[6]))
    best=min(classic,key=lambda p:(-result['totals'][p['name']+'__B6']['recovered'],
         result['totals'][p['name']+'__B6']['incorrect'],result['totals'][p['name']+'__B6']['attempts'],p['name']))
    rank_policies=[dict(name=name,policy=dict(rank='ai',rank_model=row['artifact'],gate=best['policy']['gate'],mode='sic'))
                   for name,row in models.items() if row['task']=='rank']
    state('validating_rank');result=evaluate_job(write_job('rank_selection',rank_policies,[6]))
    rows=[dict(config=row['spec']['name'],seed=row['seed'],complexity=row['complexity'],**result['totals'][name+'__B6'])
          for name,row in models.items() if row['task']=='rank']
    rank_name=select_config(rows)
    ranks={seed:models[f'rank_{safe(rank_name)}_s{seed}']['artifact'] for seed in SEEDS}
    gate_policies=[];gate_rows=[]
    for name,row in models.items():
        if row['task']!='gate':continue
        for threshold in (0,.05,.1):
            entry_name=f'{name}_t{safe(str(threshold))}'
            gate_policies.append(dict(name=entry_name,policy=dict(rank='ai',rank_model=ranks[row['seed']],gate='ai',
                gate_model=row['artifact'],gate_threshold=threshold,mode='sic')))
            gate_rows.append(dict(name=entry_name,config=f'{row["spec"]["name"]}_t{threshold}',seed=row['seed'],
                                  complexity=row['complexity'],model_name=row['spec']['name'],threshold=threshold))
    state('validating_gate');result=evaluate_job(write_job('gate_selection',gate_policies,[6]))
    scored=[dict(**r,**result['totals'][r['name']+'__B6']) for r in gate_rows];gate_config=select_config(scored)
    chosen=next(r for r in scored if r['config']==gate_config)
    selected=[]
    for seed in SEEDS:
        name=f'gate_{safe(chosen["model_name"])}_s{seed}_t{safe(str(chosen["threshold"]))}'
        selected.append(dict(name=f'ai_full_s{seed}',policy=next(p['policy'] for p in gate_policies if p['name']==name),
                             budgets=[2,4,6,8] if seed==22 else [6]))
    selected += [dict(name='classic',policy=best['policy']),
                 dict(name='ai_rank',policy=dict(rank='ai',rank_model=ranks[22],gate=best['policy']['gate'],mode='sic')),
                 dict(name='no_sic',policy={**best['policy'],'mode':'no_sic'})]
    final_job=write_job('final',selected,[2,4,6,8]);train=load_collection('train')
    model_hashes=hash_files(list((DATA/'models').glob('*'))+list((DATA/'jobs').glob('*.json')))
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),receiver_hash=receiver_hash(),source_hashes=hash_files(scientific_files()),
              model_hashes=model_hashes,selected_classical=best,selected_rank=rank_name,selected_gate=chosen,
              selected_ai='ai_full_s22',final_job=str(final_job),noise_reference=float(np.median([s['truth']['ref_power'] for s in train])),
              train_data_hashes=hash_files(sorted((DATA/'collections/train').glob('*.json'))),
              validation_hashes=hash_files([DATA/f'{name}_validation_summary.json' for name in ('classic_selection','rank_selection','gate_selection')]))
    holdouts=[r for r in make_manifest() if r['split'] in ('test','stress','noise')]
    for r in holdouts:
        r['receiver_hash']=lock['receiver_hash']
        if r['split']=='noise':r['config']['noise_reference']=lock['noise_reference'];r['config_hash']=digest(r['config'])
    lock['holdout_manifest_hashes']={s:digest([r for r in holdouts if r['split']==s]) for s in ('test','stress','noise')}
    write_immutable(DATA/'freeze.json',lock);state('frozen');return lock
