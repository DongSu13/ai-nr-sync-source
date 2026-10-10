"""Scene-grouped NR reliability experiment. Freeze on validation, then test once."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'outputs/nr_gain_v2'
FEATURES=('pss','sss','correlation','peak_ratio','consistency','abs_cfo','dmrs_power',
          'dmrs_coherence','dmrs_fit_error','dmrs_variation','llr_abs','llr_low_fraction','estimated_noise')
LOG_FEATURES=('peak_ratio','abs_cfo','dmrs_power','dmrs_variation','llr_abs','estimated_noise')
CLASSICAL=('correlation','sss','dmrs_coherence','llr_abs','dmrs_fit','joint_pilot')

def validate_features(f):
    if set(f)!=set(FEATURES): raise ValueError('Unexpected/missing feature; possible truth leakage')
    if not np.isfinite([f[k] for k in FEATURES]).all(): raise ValueError('Nonfinite measured features')

def load_split(split,expected):
    paths=sorted((DATA/split).glob('*.json'))
    if len(paths)!=expected: raise ValueError(f'{split}: expected {expected} complete scenes, got {len(paths)}')
    rows=[]; labels=[]; groups=[]; rounds=[]; pcis=[]; channels=[]; snrs=[]; truths={}; digest=hashlib.sha256();seen=set()
    for p in paths:
        content=p.read_bytes();digest.update(content);s=json.loads(content)
        if s['split']!=split or s['seed'] in seen:raise ValueError('Duplicate or misplaced scene')
        seen.add(s['seed']); truths[s['seed']]=len(s['users']) if isinstance(s['users'],list) else 1
        for c in s['candidates']:
            validate_features(c['features']);rows.append([c['features'][k] for k in FEATURES]);labels.append(c['label'])
            groups.append(s['seed']);rounds.append(c['round']);pcis.append(c['pci']);channels.append(s['channel']);snrs.append(s['snr_db'])
    return dict(x=np.asarray(rows,float),y=np.asarray(labels,int),groups=np.asarray(groups),rounds=np.asarray(rounds),
                pcis=np.asarray(pcis),channels=np.asarray(channels),snrs=np.asarray(snrs),truths=truths,sha256=digest.hexdigest(),scenes=len(paths))

def transformed(x):
    z=x.copy()
    for name in LOG_FEATURES:
        i=FEATURES.index(name);z[:,i]=np.log1p(np.maximum(z[:,i],0))
    return z

def classical_scores(x,name):
    if name in FEATURES:return x[:,FEATURES.index(name)]
    if name=='dmrs_fit':return -x[:,FEATURES.index('dmrs_fit_error')]
    if name=='joint_pilot':return x[:,FEATURES.index('sss')]*np.sqrt(np.maximum(x[:,FEATURES.index('dmrs_coherence')],0))
    raise ValueError(name)

def threshold_at_fpr(scores,y,target=.10):
    scores=np.asarray(scores);y=np.asarray(y)
    if set(y)!={0,1}:raise ValueError('Both classes required')
    order=np.argsort(-scores,kind='stable');v=scores[order];lab=y[order]
    end=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1]
    tp=np.cumsum(lab)[end];fp=np.cumsum(1-lab)[end]
    eligible=np.flatnonzero(fp/(y==0).sum()<=target)
    if not len(eligible):return float(np.nextafter(v[0],np.inf))
    # Maximize positives, then minimize negatives, then use the higher threshold.
    best=max(eligible,key=lambda j:(tp[j],-fp[j],v[end[j]]))
    return float(v[end[best]])

def rates(scores,y,threshold):
    pred=np.asarray(scores)>=threshold;y=np.asarray(y)
    tp=int(np.sum(pred & (y==1)));fp=int(np.sum(pred & (y==0)));tn=int(np.sum(~pred & (y==0)));fn=int(np.sum(~pred & (y==1)))
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,tpr=tp/max(tp+fn,1),fpr=fp/max(fp+tn,1),threshold=float(threshold),
                average_precision=float(average_precision_score(y,scores)),auc=float(roc_auc_score(y,scores)))

def paired_ci(ai,baseline,y,groups,reps=2000):
    ids=np.unique(groups);counts=[]
    for g in ids:
        m=groups==g;p=m&(y==1);n=m&(y==0)
        counts.append([np.sum(ai&p)-np.sum(baseline&p),p.sum(),np.sum(ai&n)-np.sum(baseline&n),n.sum()])
    a=np.asarray(counts,float);rng=np.random.default_rng(95341);out=[]
    for _ in range(reps):
        total=a[rng.integers(len(a),size=len(a))].sum(0)
        out.append([total[0]/max(total[1],1),total[2]/max(total[3],1)])
    q=np.quantile(out,[.025,.975],axis=0)
    return dict(delta_tpr_ci95=q[:,0].tolist(),delta_fpr_ci95=q[:,1].tolist(),bootstrap_scenes=len(ids),repetitions=reps)

def budget_metrics(scores,d,budget):
    recovered=0;eligible=0
    # First-pass ranking: no preceding CRC/SIC can leak information into it.
    for g in np.unique(d['groups']):
        ix=np.flatnonzero((d['groups']==g)&(d['rounds']==0));order=ix[np.argsort(-scores[ix],kind='stable')]
        chosen=[];seen=set()
        for j in order:
            if int(d['pcis'][j]) in seen:continue
            seen.add(int(d['pcis'][j]));chosen.append(j)
            if len(chosen)==budget:break
        recovered+=len(set(d['pcis'][chosen][d['y'][chosen]==1]))
        eligible+=len(set(d['pcis'][ix][d['y'][ix]==1]))
    return dict(budget_per_scene=budget,confirmed_users=recovered,available_confirmable_users=eligible,
                users_per_scene=recovered/d['scenes'],fraction_of_available=recovered/max(eligible,1))

def freeze():
    if (DATA/'frozen.joblib').exists():raise RuntimeError('Already frozen; preserve original holdout protocol')
    train=load_split('train',500);val=load_split('validation',150)
    if set(train['groups'])&set(val['groups']):raise ValueError('Group leakage')
    x=transformed(train['x']);v=transformed(val['x']);y=val['y'];models={};thresholds={};validation={}
    for name in CLASSICAL:
        score=classical_scores(val['x'],name);t=threshold_at_fpr(score,y);thresholds[name]=t;validation[name]=rates(score,y,t)
    for c in [.1,1.,10.]:
        name=f'logistic_C{c}';model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=22))
        model.fit(x,train['y']);score=model.predict_proba(v)[:,1];t=threshold_at_fpr(score,y)
        models[name]=model;thresholds[name]=t;validation[name]=rates(score,y,t)
    for leaves in [7,15]:
        name=f'hgb_leaves{leaves}';model=HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=150,learning_rate=.06,
                min_samples_leaf=30,l2_regularization=1.,early_stopping=False,random_state=22)
        model.fit(x,train['y']);score=model.predict_proba(v)[:,1];t=threshold_at_fpr(score,y)
        models[name]=model;thresholds[name]=t;validation[name]=rates(score,y,t)
    key=lambda name:(validation[name]['tpr'],-validation[name]['fpr'],validation[name]['average_precision'])
    best_ai=max(models,key=key);best_classic=max(CLASSICAL,key=key)
    manifest=dict(features=FEATURES,train_hash=train['sha256'],validation_hash=val['sha256'],thresholds=thresholds,
        selected_ai=best_ai,selected_classical=best_classic,validation=validation,train_scenes=500,validation_scenes=150,
        test_scenes=800,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'src/matlab/nr_candidate_scene.m',ROOT/'src/matlab/run_nr_campaign.m']})
    joblib.dump(dict(models=models,manifest=manifest),DATA/'frozen.joblib')
    (DATA/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected_ai=best_ai,selected_classical=best_classic,validation=validation),indent=2))

def evaluate(split):
    output=DATA/f'{split}_evaluation.json'
    if output.exists():raise RuntimeError('Evaluation already recorded; no silent overwrites')
    frozen=joblib.load(DATA/'frozen.joblib');m=frozen['manifest'];d=load_split(split,800 if split=='test' else 150)
    scores={n:classical_scores(d['x'],n) for n in CLASSICAL}
    scores.update({n:model.predict_proba(transformed(d['x']))[:,1] for n,model in frozen['models'].items()})
    results={n:rates(s,d['y'],m['thresholds'][n]) for n,s in scores.items()}
    ai=m['selected_ai'];baseline=m['selected_classical'];delta=results[ai]['tpr']-results[baseline]['tpr'];fd=results[ai]['fpr']-results[baseline]['fpr']
    ci=paired_ci(scores[ai]>=m['thresholds'][ai],scores[baseline]>=m['thresholds'][baseline],d['y'],d['groups'])
    subsets={}
    for channel in ['AWGN','TDL-C']:
        for snr in [-6,0,6,12]:
            ix=(d['channels']==channel)&(d['snrs']==snr)
            subsets[f'{channel}_{snr}']={n:rates(scores[n][ix],d['y'][ix],m['thresholds'][n]) for n in [ai,baseline,'correlation']}
    result=dict(split=split,scenes=d['scenes'],candidates=len(d['y']),positive=int(d['y'].sum()),data_hash=d['sha256'],
        selected_ai=ai,selected_classical=baseline,methods=results,delta_tpr=delta,delta_fpr=fd,paired_bootstrap=ci,
        gain_criterion_pass=bool(delta>=.05 and fd<=.02 and ci['delta_tpr_ci95'][0]>0),subsets=subsets,
        first_pass_budget={n:[budget_metrics(s,d,k) for k in [1,2]] for n,s in scores.items()})
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(DATA/f'{split}_predictions.npz',y=d['y'],groups=d['groups'],rounds=d['rounds'],pcis=d['pcis'],**scores)
    print(json.dumps({k:result[k] for k in ['split','scenes','candidates','selected_ai','selected_classical','methods','delta_tpr','delta_fpr','paired_bootstrap','gain_criterion_pass']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','development','test']);args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:evaluate(args.action)
