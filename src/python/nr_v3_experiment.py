"""Train on development data, freeze once, evaluate untouched scene suites."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import sklearn
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from nr_v3_campaign import ROOT, DATA, config_hash, receiver_hash
from nr_v3_data import FEATURES, load_scenes, arrays
from nr_v3_metrics import budget_counts, summarize_counts, paired_user_ci, threshold_at_fpr, candidate_metrics
from nr_v3_models import CLASSICAL, model_specs, make_model, classical_scores

SEEDS=(22,23,24)
TARGETS=(.01,.05,.1)
COUNTS=dict(smoke=64,train=3000,validation=1000,development=500,test=2000,sweep=3200,stress=800,noise=600)
ABLATIONS=dict(no_correlation={'pss','sss','correlation','peak_ratio','consistency'},
               no_dmrs={'dmrs_power','dmrs_coherence','dmrs_fit_error','dmrs_variation'},
               no_llr={'llr_abs','llr_low_fraction'})


def hash_files(paths):
    return {str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def check_hashes(hashes):
    for path,expected in hashes.items():
        if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'file changed since freeze: {path}')


def write_new_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,indent=2,allow_nan=False)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    # NTFS hard-link creation is atomic and refuses to replace an existing file.
    os.link(temporary,path);temporary.unlink()


def write_predictions(path,values):
    path=Path(path)
    if path.exists():
        with np.load(path) as saved:
            if set(saved.files)!=set(values) or any(not np.array_equal(saved[k],v) for k,v in values.items()):
                raise ValueError('different existing predictions')
        return
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**values);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);temporary.unlink()


def assert_disjoint(*sets):
    used=set()
    for scenes in sets:
        ids={int(s['seed']) for s in scenes}
        if len(ids)!=len(scenes) or used & ids:raise ValueError('scene overlap')
        used |= ids


def select_winners(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['family'],row['name'])].append(row)
    families=defaultdict(list)
    for (family,name), group in grouped.items():
        if sorted(r['seed'] for r in group)!=list(SEEDS):raise ValueError('missing or duplicate training seed')
        families[family].append((np.mean([r['recovery'] for r in group]),np.mean([r['ap'] for r in group]),name))
    return {f:sorted(v,key=lambda t:(-t[0],-t[1],t[2]))[0][2] for f,v in families.items()}


def scientific_hashes():
    names=['nr_v3_campaign.py','nr_v3_data.py','nr_v3_metrics.py','nr_v3_models.py','nr_v3_experiment.py','run_nr_v3_pipeline.py']
    return hash_files([ROOT/'src/python'/n for n in names])


def split_scenes(split,root=DATA):
    root=Path(root)
    records=json.loads((root/'manifests'/f'{split}.json').read_text())
    if len(records)!=COUNTS[split]:raise ValueError(f'incomplete {split} manifest')
    paths=sorted((root/'scenes'/split).rglob('*.json'))
    if len(paths)!=len(records):raise ValueError(f'{split}: {len(paths)}/{len(records)} complete scenes')
    scenes=load_scenes(paths);expected={r['seed']:r for r in records};sha=receiver_hash()
    digest=hashlib.sha256()
    for p,s in zip(paths,scenes):
        r=expected.get(s['seed'])
        if r is None or any(s[k]!=r[k] for k in ('split','suite','cell','config','config_hash','receiver_hash')):
            raise ValueError(f'manifest mismatch: {p}')
        if s['config_hash']!=config_hash(s['config']) or s['receiver_hash']!=sha:raise ValueError('scene hash mismatch')
        channel_seeds=[int(u['channel_seed']) for u in s['users']]
        if len(set(channel_seeds))!=len(channel_seeds):raise ValueError('shared user channel seed')
        if len(s['users'])!=s['config']['nu']:raise ValueError('wrong transmitted user count')
        if s['noise_variance']<=0:raise ValueError('zero noise variance')
        if any(c['features']['dmrs_power']<=0 for c in s['candidates']):raise ValueError('unmeasured candidate DMRS')
        digest.update(p.read_bytes())
    return scenes,digest.hexdigest()


def validation_summary(scores,d):
    return dict(recovery=summarize_counts(budget_counts(scores,d,2))['recovery_rate'],
                ap=candidate_metrics(scores,d['y'],float('inf'))['AP'])


def fit_one(spec,seed,train,val,recipe,variant='full',keep=FEATURES,root=DATA):
    folder=Path(root)/'models';folder.mkdir(parents=True,exist_ok=True)
    stem=f"{variant}__{spec['name']}__s{seed}";meta_path=folder/(stem+'.json');model_path=folder/(stem+'.joblib')
    expected=config_hash(dict(recipe=recipe,spec=spec,seed=seed,keep=list(keep)))
    if meta_path.exists():
        row=json.loads(meta_path.read_text())
        if row['recipe_hash']!=expected:raise ValueError('training cache recipe changed')
        check_hashes(row['model_hash']);return row
    model=make_model(spec,seed,tuple(keep));t=time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded,threadpool_limits(limits=2):
        warnings.simplefilter('always');model.fit(train['x'],train['y'])
    elapsed=time.perf_counter()-t
    with threadpool_limits(limits=2):scores=model.predict_proba(val['x'])[:,1]
    row=dict(name=spec['name'],family=spec['family'],seed=seed,variant=variant,recipe_hash=expected,
             training_seconds=elapsed,warnings=[str(w.message) for w in recorded],artifact=str(model_path.resolve()),
             **validation_summary(scores,val))
    if model_path.exists():raise ValueError(f'orphan model requires review before reuse: {model_path}')
    tmp=model_path.with_suffix('.joblib.tmp');joblib.dump(model,tmp);tmp.rename(model_path)
    row['model_hash']=hash_files([model_path]);write_new_json(meta_path,row)
    print(f"FIT {stem} seconds={elapsed:.1f} recovery={row['recovery']:.4f} AP={row['ap']:.4f} warnings={len(row['warnings'])}",flush=True)
    return row


def train(root=DATA):
    root=Path(root)
    if (root/'freeze.json').exists():raise ValueError('training already frozen')
    ts,th=split_scenes('train',root);vs,vh=split_scenes('validation',root);assert_disjoint(ts,vs)
    t,v=arrays(ts),arrays(vs);recipe=dict(train_hash=th,validation_hash=vh,source_hashes=scientific_hashes(),
                                      sklearn=sklearn.__version__,receiver_hash=receiver_hash())
    rows=[fit_one(spec,seed,t,v,recipe,root=root) for spec in model_specs() for seed in SEEDS]
    winners=select_winners(rows)
    selected=sorted(winners.values(),key=lambda n:(-np.mean([r['recovery'] for r in rows if r['name']==n]),
                                                  -np.mean([r['ap'] for r in rows if r['name']==n]),n))[0]
    family=next(r['family'] for r in rows if r['name']==selected)
    ablations=[]
    for variant,removed in ABLATIONS.items():
        keep=tuple(f for f in FEATURES if f not in removed)
        for spec in model_specs():
            if spec['family']==family:
                for seed in SEEDS:ablations.append(fit_one(spec,seed,t,v,recipe,variant,keep,root))
    summary=dict(recipe=recipe,rows=rows,family_winners=winners,selected_ai=selected,ablations=ablations)
    path=root/'training_summary.json'
    if path.exists():
        if json.loads(path.read_text())!=summary:raise ValueError('different training summary')
    else:write_new_json(path,summary)
    return summary


def freeze(root=DATA):
    root=Path(root)
    if (root/'freeze.json').exists():raise FileExistsError('freeze already exists')
    summary=json.loads((root/'training_summary.json').read_text());check_hashes(summary['recipe']['source_hashes'])
    ts,th=split_scenes('train',root);vs,vh=split_scenes('validation',root);ds,_=split_scenes('development',root)
    assert_disjoint(ts,vs,ds)
    if th!=summary['recipe']['train_hash'] or vh!=summary['recipe']['validation_hash']:raise ValueError('training data changed')
    v=arrays(vs);methods={};model_paths={}
    for name in CLASSICAL:methods[name]=classical_scores(v['x'],name)
    selected_rows=[r for r in summary['rows'] if r['name'] in summary['family_winners'].values()]
    for variant in ABLATIONS:
        rows=[r for r in summary['ablations'] if r['variant']==variant];best=next(iter(select_winners(rows).values()))
        selected_rows += [r for r in rows if r['name']==best]
    for row in selected_rows:
        name=f"{row['variant']}__{row['name']}__s{row['seed']}";check_hashes(row['model_hash'])
        model_paths[name]=row['artifact']
        with threadpool_limits(limits=2):methods[name]=joblib.load(row['artifact']).predict_proba(v['x'])[:,1]
    classic_validation={n:validation_summary(methods[n],v) for n in CLASSICAL}
    classic=sorted(CLASSICAL,key=lambda n:(-classic_validation[n]['recovery'],-classic_validation[n]['ap'],n))[0]
    thresholds={name:{str(p):threshold_at_fpr(scores,v['y'],p) for p in TARGETS} for name,scores in methods.items()}
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_ai=f"full__{summary['selected_ai']}__s22",
              selected_classical=classic,family_winners=summary['family_winners'],model_paths=model_paths,thresholds=thresholds,
              source_hashes=scientific_hashes(),receiver_hash=receiver_hash(),model_hashes=hash_files(model_paths.values()),
              train_hash=th,validation_hash=vh,noise_reference=float(np.median([s['ref_power'] for s in ts])),
              sklearn_version=sklearn.__version__,numpy_version=np.__version__,classic_validation=classic_validation,
              primary_endpoint='first-pass B=2 recovered users / all transmitted users; IID test only')
    write_new_json(root/'freeze.json',lock);print(json.dumps({k:lock[k] for k in ['selected_ai','selected_classical','noise_reference']},indent=2),flush=True)
    return lock


def score_methods(d,lock):
    scores={};times={}
    for name in CLASSICAL:
        t=time.perf_counter();scores[name]=classical_scores(d['x'],name);times[name]=time.perf_counter()-t
    for name,path in lock['model_paths'].items():
        model=joblib.load(path);t=time.perf_counter()
        with threadpool_limits(limits=2):scores[name]=model.predict_proba(d['x'])[:,1]
        times[name]=time.perf_counter()-t
    return scores,times


def evaluate_suite(split,root=DATA):
    root=Path(root);out=root/f'{split}_evaluation.json'
    if out.exists():raise FileExistsError(out)
    lock=json.loads((root/'freeze.json').read_text());check_hashes(lock['source_hashes']);check_hashes(lock['model_hashes'])
    if receiver_hash()!=lock['receiver_hash']:raise ValueError('receiver changed')
    scenes,digest=split_scenes(split,root);d=arrays(scenes);scores,times=score_methods(d,lock)
    methods={};cell_results={};cell_comparisons={};saved_counts={}
    crc=np.array([c['crc'] for s in scenes for c in s['candidates']],bool)
    for name,values in scores.items():
        budgets={}
        for b in (1,2,4,6):
            counts=budget_counts(values,d,b);budgets[str(b)]=summarize_counts(counts)
            if b==2:saved_counts[name]=counts
        methods[name]=dict(budgets=budgets,candidate={str(p):candidate_metrics(values,d['y'],lock['thresholds'][name][str(p)]) for p in TARGETS})
        methods[name]['crc_after_candidate_filter']={str(p):dict(
            crc_outputs=int(np.sum(crc & (values>=lock['thresholds'][name][str(p)]))),
            incorrect_outputs=int(np.sum(crc & (d['y']==0) & (values>=lock['thresholds'][name][str(p)])))) for p in TARGETS}
    # Auxiliary comparisons use the same scenes, not individually resampled hypotheses.
    ai,base=lock['selected_ai'],lock['selected_classical']
    for name,counts in saved_counts.items():
        methods[name]['exploratory_vs_classical']=paired_user_ci(counts,saved_counts[base])
    primary=paired_user_ci(saved_counts[ai],saved_counts[base])
    primary['confirmatory']=split=='test'
    primary['positive_gain_supported']=split=='test' and primary['ci95'][0] is not None and primary['ci95'][0]>0
    for cell in sorted({s['cell'] for s in scenes}):
        subset=[s for s in scenes if s['cell']==cell];ids={s['seed'] for s in subset}
        mask=np.isin(d['groups'],list(ids));sd=arrays(subset)
        # arrays(subset) preserves original scene order, as does the mask.
        cell_results[cell]={}
        base_counts=budget_counts(scores[base][mask],sd,2)
        for name,values in scores.items():
            counts=budget_counts(values[mask],sd,2)
            cell_results[cell][name]=dict(budget2=summarize_counts(counts),
                candidate=candidate_metrics(values[mask],sd['y'],lock['thresholds'][name]['0.01']),
                exploratory_vs_classical=paired_user_ci(counts,base_counts))
        cell_comparisons[cell]=paired_user_ci(budget_counts(scores[ai][mask],sd,2),budget_counts(scores[base][mask],sd,2))
    noise=None
    if split=='noise':
        events=sum(any(c['crc'] for c in s['candidates']) for s in scenes);n=len(scenes)
        noise=dict(scenes=n,scenes_with_any_crc_output=events,scope='all 12 offline candidates, before method filtering',
                   upper95=float(beta.ppf(.95,events+1,n-events)) if events<n else 1.)
    timing={k:float(np.median([s['timings'][k] for s in scenes])) for k in scenes[0]['timings']}
    result=dict(split=split,scenes=len(scenes),candidates=len(d['y']),positive_candidates=int(d['y'].sum()),
        data_hash=digest,selected_ai=ai,selected_classical=base,primary=primary,methods=methods,cells=cell_results,
        score_seconds=times,median_scene_timing=timing,noise=noise,exploratory_cell_comparisons=cell_comparisons,
        timing_scope='offline feature extraction/all-candidate labeling measured; scoring batch measured; no online speedup claim')
    predictions=root/f'{split}_predictions.npz'
    write_predictions(predictions,dict(y=d['y'],groups=d['groups'],pcis=d['pcis'],scene_ids=d['scene_ids'],**scores))
    write_new_json(out,result);print(f'EVALUATED {split}: {primary}',flush=True)
    return result


def independent_recovery(scores,d,budget):
    recovered=0;attempts=0;transmitted=0
    for scene_id in d['scene_ids']:
        indexes=[int(i) for i in np.flatnonzero(d['groups']==scene_id)]
        indexes.sort(key=lambda i:(-float(scores[i]),i));seen=set();chosen=[]
        for index in indexes:
            pci=int(d['pcis'][index])
            if pci not in seen and len(chosen)<budget:
                seen.add(pci);chosen.append(index)
        recovered+=sum(bool(d['y'][i]) for i in chosen)
        attempts+=len(chosen);transmitted+=d['true_users'][int(scene_id)]
    return recovered,transmitted,attempts


def audit(root=DATA):
    root=Path(root);lock=json.loads((root/'freeze.json').read_text());check_hashes(lock['source_hashes']);check_hashes(lock['model_hashes'])
    all_scenes=[];rows={}
    for split in COUNTS:
        if split=='smoke':continue
        scenes,digest=split_scenes(split,root);assert_disjoint(all_scenes,scenes);all_scenes+=scenes
        rows[split]=dict(scenes=len(scenes),data_hash=digest)
        if split in ('test','sweep','stress','noise'):
            result=json.loads((root/f'{split}_evaluation.json').read_text())
            if result['data_hash']!=digest:raise ValueError('evaluated data modified')
            predicted=np.load(root/f'{split}_predictions.npz');d=arrays(scenes)
            if not np.array_equal(predicted['y'],d['y']):raise ValueError('prediction label mismatch')
            # Independent direct confusion-count verification for every saved method/threshold.
            for name in lock['thresholds']:
                for budget in (1,2,4,6):
                    actual=independent_recovery(predicted[name],d,budget)
                    expected=tuple(result['methods'][name]['budgets'][str(budget)][k] for k in
                                   ('recovered_users','transmitted_users','decode_attempts'))
                    if actual!=expected:raise ValueError('independent user recovery recount mismatch')
                for target,threshold in lock['thresholds'][name].items():
                    positive=predicted[name]>=threshold;y=d['y'].astype(bool)
                    actual=[int(np.sum(positive&y)),int(np.sum(positive&~y)),int(np.sum(~positive&~y)),int(np.sum(~positive&y))]
                    expected=[result['methods'][name]['candidate'][target][k] for k in ('tp','fp','tn','fn')]
                    if actual!=expected:raise ValueError('independent confusion recount mismatch')
    if rows['train']['data_hash']!=lock['train_hash'] or rows['validation']['data_hash']!=lock['validation_hash']:
        raise ValueError('frozen training data mismatch')
    output=dict(status='PASS',unique_scenes=len(all_scenes),splits=rows,receiver_hash=receiver_hash())
    write_new_json(root/'integrity_audit.json',output);print('AUDIT PASS',len(all_scenes),flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','freeze','evaluate','audit']);p.add_argument('--split',choices=['test','sweep','stress','noise']);a=p.parse_args()
    if a.action=='train':train()
    elif a.action=='freeze':freeze()
    elif a.action=='audit':audit()
    elif a.split:evaluate_suite(a.split)
    else:p.error('--split is required for evaluate')
