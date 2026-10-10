"""Deterministic v4 manifests, isolated seeds and freeze-gated holdouts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'outputs/nr_gain_v4'
COUNTS=dict(train=1200,validation=400,development=200,test=2000,stress=800,noise=200)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def receiver_hash():
    paths=sorted((ROOT/'src/matlab').glob('nr_v4_*.m'))+[ROOT/'src/matlab/nr_v3_grid.m']
    h=hashlib.sha256()
    for p in paths:h.update(p.name.encode()+b'\0'+p.read_bytes()+b'\0')
    return h.hexdigest()


def anchor():
    return dict(nu=3,snr_db=0,profile='TDL-C',spread_s=300e-9,doppler=30,
                cfo_max=500,gap_db=6,timing_mode='random',pss_mode='random',noise_scale=1)


def record(seed,split,cell,cfg):
    return dict(schema='nr_gain_v4',seed=seed,split=split,cell=cell,config=cfg,config_hash=digest(cfg))


def make_manifest():
    rng=np.random.default_rng(20260923);rows=[]
    levels=dict(nu=[2,3,4],snr_db=[-6,-3,0,3,6],profile=['AWGN','TDL-A','TDL-C'],
                gap_db=[0,6,12],spread_s=[100e-9,300e-9],doppler=[0,30,100],pss_mode=['same','random'])
    for j,split in enumerate(('train','validation','development','test')):
        for i in range(COUNTS[split]):
            cfg=anchor()
            for key,values in levels.items():cfg[key]=values[int(rng.integers(len(values)))]
            rows.append(record(40000001+j*100000+i,split,'mixed',cfg))
    changes=[dict(nu=6,snr_db=-6,gap_db=18),dict(pss_mode='same',timing_mode='near'),
             dict(profile='TDL-D',doppler=300),dict(spread_s=1000e-9,cfo_max=1500)]
    for j,change in enumerate(changes):
        for i in range(200):rows.append(record(40400001+j*200+i,'stress',f'stress_{j+1}',{**anchor(),**change}))
    for i in range(200):rows.append(record(40500001+i,'noise','noise_1',{**anchor(),'nu':0}))
    return rows


def smoke_manifest():
    profiles=[dict(nu=1,snr_db=100,profile=p,cfo_max=0,doppler=0) for p in ('AWGN','TDL-A','TDL-C')]
    profiles += [dict(nu=2,snr_db=6,profile='AWGN',gap_db=12,pss_mode='different'),
                 dict(nu=2,snr_db=6,gap_db=12,pss_mode='same'),
                 dict(nu=3,snr_db=6,pss_mode='same',timing_mode='near',cfo_max=1500),
                 dict(nu=0,noise_reference=.01),dict(nu=6,snr_db=0,profile='TDL-D',gap_db=18)]
    return [record(49000001+j*4+i,'smoke',f'smoke_{j+1}',{**anchor(),**change})
            for j,change in enumerate(profiles) for i in range(4)]


def write_immutable(path,value):
    path=Path(path);content=json.dumps(value,indent=2,allow_nan=False)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'different existing artifact: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(content,encoding='utf-8');temporary.rename(path)


def prepare(root=DATA,stage='smoke'):
    root=Path(root)
    if stage not in ('smoke','development','holdout'):raise ValueError('invalid stage')
    if stage=='holdout' and not (root/'freeze.json').exists():raise ValueError('holdout requires freeze')
    rows=smoke_manifest() if stage=='smoke' else make_manifest()
    keep={'smoke'} if stage=='smoke' else {'train','validation','development'} if stage=='development' else {'test','stress','noise'}
    rows=[r for r in rows if r['split'] in keep];sha=receiver_hash()
    if stage=='holdout':
        frozen=json.loads((root/'freeze.json').read_text())
        if frozen['receiver_hash']!=sha:raise ValueError('receiver differs from freeze')
        pref=float(frozen['noise_reference'])
        if not np.isfinite(pref) or pref<=0:raise ValueError('invalid frozen noise reference')
        for r in rows:
            if r['split']=='noise':r['config']['noise_reference']=pref;r['config_hash']=digest(r['config'])
    for r in rows:r['receiver_hash']=sha
    if stage=='holdout':
        for split in keep:
            if digest([r for r in rows if r['split']==split])!=frozen['holdout_manifest_hashes'][split]:
                raise ValueError('holdout manifest differs from freeze')
    for split in sorted(keep):write_immutable(root/'manifests'/f'{split}.json',[r for r in rows if r['split']==split])
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','development','holdout']);a=p.parse_args()
    print(a.stage,len(prepare(stage=a.stage)))
