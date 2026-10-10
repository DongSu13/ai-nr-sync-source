"""Small train-only pipelines and transparent numeric exports for MATLAB."""
from __future__ import annotations
import argparse
import json
import numpy as np
from scipy.special import expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from nr_v4_campaign import DATA,write_immutable


def specs(task):
    if task not in ('rank','gate'):raise ValueError('unknown prediction task')
    linear='logistic' if task=='rank' else 'ridge'
    values=(.01,.1,1) if task=='rank' else (.1,1,10)
    return [dict(task=task,kind=linear,param=v,name=f'{linear}_{v}') for v in values]+[
        dict(task=task,kind='hgb',param=v,name=f'hgb_{v}') for v in (7,15,31)]


def make_model(spec,seed):
    if spec not in specs(spec['task']):raise ValueError('unknown model specification')
    if spec['kind']=='logistic':est=LogisticRegression(C=spec['param'],max_iter=2000,random_state=seed)
    elif spec['kind']=='ridge':est=Ridge(alpha=spec['param'])
    else:
        cls=HistGradientBoostingClassifier if spec['task']=='rank' else HistGradientBoostingRegressor
        est=cls(max_leaf_nodes=spec['param'],learning_rate=.1,max_iter=150,early_stopping=False,random_state=seed)
    return Pipeline([('scale',StandardScaler()),('model',est)])


def export_model(model,task):
    scale=model.named_steps['scale'];est=model.named_steps['model']
    out=dict(task=task,width=int(scale.n_features_in_),mean=scale.mean_.tolist(),scale=scale.scale_.tolist())
    if isinstance(est,(LogisticRegression,Ridge)):
        out.update(kind='linear',coef=np.asarray(est.coef_).reshape(-1).tolist(),intercept=float(np.asarray(est.intercept_).reshape(-1)[0]))
    else:
        trees=[]
        for stage in est._predictors:
            assert len(stage)==1,'Only binary/rank or scalar/gate supported'
            nodes=[]
            for n in stage[0].nodes:
                if n['is_leaf']:nodes.append(dict(leaf=True,value=float(n['value'])))
                else:
                    assert not n['is_categorical']
                    nodes.append(dict(leaf=False,feature=int(n['feature_idx']),threshold=float(n['num_threshold']),left=int(n['left']),right=int(n['right'])))
            trees.append(nodes)
        out.update(kind='trees',baseline=float(est._baseline_prediction[0,0]),trees=trees)
    return out


def predict_export(artifact,x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=artifact['width'] or not np.isfinite(x).all():raise ValueError('bad predictor input')
    z=(x-np.asarray(artifact['mean']))/np.asarray(artifact['scale'])
    if artifact['kind']=='linear':raw=z@np.asarray(artifact['coef'])+artifact['intercept']
    else:
        raw=np.full(len(z),artifact['baseline'],dtype=float)
        for tree in artifact['trees']:
            for i,row in enumerate(z):
                n=tree[0]
                while not n['leaf']:n=tree[n['left'] if row[n['feature']]<=n['threshold'] else n['right']]
                raw[i]+=n['value']
    return expit(raw) if artifact['task']=='rank' else raw


def parity_fixture():
    rng=np.random.default_rng(63);fixtures=[]
    for task in ('rank','gate'):
        width=17 if task=='rank' else 21;x=rng.normal(size=(120,width));z=rng.normal(size=(11,width))
        y=(x[:,0]>0).astype(int) if task=='rank' else -x[:,0]+.2*x[:,1]
        for spec in specs(task):
            with threadpool_limits(1):
                model=make_model(spec,22).fit(x,y)
                expected=model.predict_proba(z)[:,1] if task=='rank' else model.predict(z)
                artifact=export_model(model,task)
                np.testing.assert_allclose(predict_export(artifact,z),expected,atol=1e-10,rtol=1e-10)
            fixtures.append(dict(name=spec['name'],artifact=artifact,x=z.tolist(),expected=expected.tolist()))
    write_immutable(DATA/'predictor_parity.json',fixtures)
    print('Exported 12 cross-language predictor parity fixtures')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['parity']);p.parse_args();parity_fixture()
