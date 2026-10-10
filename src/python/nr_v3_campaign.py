"""Immutable scene manifests; final manifests are inaccessible before freeze."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs/nr_gain_v3'
RECEIVER_FILES = ('nr_v3_grid.m', 'nr_v3_search.m', 'nr_v3_scene.m', 'run_nr_v3.m', 'nr_v3_receiver_hash.m')


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def receiver_hash():
    digest = hashlib.sha256()
    for name in RECEIVER_FILES:
        digest.update(name.encode() + b'\0' + (ROOT / 'src/matlab' / name).read_bytes() + b'\0')
    return digest.hexdigest()


def anchor():
    return dict(nu=3, snr_db=0, profile='TDL-C', spread_s=300e-9, doppler=30,
                cfo_max=500, gap_db=6, timing_mode='random', pss_mode='random', noise_scale=1)


def record(seed, split, cell, config):
    return dict(schema='nr_gain_v3', seed=seed, split=split, suite=split, cell=cell,
                config=config, config_hash=config_hash(config))


def make_manifest():
    rng = np.random.default_rng(20260922)
    out = []
    for si, (split, count) in enumerate([('train',3000),('validation',1000),('development',500),('test',2000)]):
        for i in range(count):
            cfg = anchor()
            for key, values in dict(nu=[1,2,3,4], snr_db=list(range(-9,13,3)),
                                    profile=['AWGN','TDL-A','TDL-C'], gap_db=[0,6,12],
                                    spread_s=[100e-9,300e-9], doppler=[0,30,100]).items():
                cfg[key] = values[int(rng.integers(len(values)))]
            out.append(record(10000000 + si*100000+i+1, split, 'mixed', cfg))
    sweeps = dict(snr_db=list(range(-12,13,3)), nu=[1,2,3,4,6], gap_db=[0,6,12,18],
                  profile=['AWGN','TDL-A','TDL-C','TDL-D'], doppler=[0,30,300],
                  cfo_max=[500,1000,1500], pss_mode=['same','different'], timing_mode=['near','separated'])
    index = 0
    for key, values in sweeps.items():
        for value in values:
            for _ in range(100):
                index += 1
                out.append(record(10400000+index,'sweep',f'{key}_{value}',{**anchor(), key:value}))
    stress = [dict(nu=6,snr_db=-6,gap_db=18), dict(pss_mode='same',timing_mode='near'),
              dict(profile='TDL-D',doppler=300), dict(spread_s=1000e-9,cfo_max=1500)]
    for j, changes in enumerate(stress):
        for i in range(200):
            out.append(record(10500001+j*200+i,'stress',f'stress_{j+1}',{**anchor(),**changes}))
    for j, scale in enumerate([.25,1,4]):
        for i in range(200):
            out.append(record(10600001+j*200+i,'noise',f'noise_{scale}',{**anchor(),'nu':0,'noise_scale':scale}))
    return out


def smoke_manifest():
    final = make_manifest()
    selected = [r for r in final if r['split'] in ('sweep','stress','noise')][::100]
    selected += [r for r in final if r['split']=='train'][:64-len(selected)]
    out=[]
    for i,r in enumerate(selected):
        cfg={**r['config']}
        if cfg['nu']==0:cfg['noise_reference']=0.01
        out.append(record(19000001+i,'smoke',r['cell'],cfg))
    return out


def write_manifest(path, records):
    path = Path(path)
    content = json.dumps(records, indent=2, allow_nan=False)
    if path.exists():
        if json.loads(path.read_text()) != records:raise ValueError(f'different existing manifest: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(content)


def prepare(root=DATA, stage='development'):
    root=Path(root)
    if stage=='holdout' and not (root/'freeze.json').exists():
        raise ValueError('holdout requires freeze.json')
    if stage=='smoke':records=smoke_manifest()
    else:
        keep={'train','validation','development'} if stage=='development' else {'test','sweep','stress','noise'}
        records=[r for r in make_manifest() if r['split'] in keep]
    sha=receiver_hash()
    if stage=='holdout':
        frozen=json.loads((root/'freeze.json').read_text())
        if frozen['receiver_hash']!=sha:raise ValueError('receiver changed since freeze')
        pref=float(frozen['noise_reference'])
        if not np.isfinite(pref) or pref<=0:raise ValueError('invalid noise reference')
        for r in records:
            if r['split']=='noise':
                r['config']['noise_reference']=pref
                r['config_hash']=config_hash(r['config'])
    for r in records:r['receiver_hash']=sha
    for split in sorted({r['split'] for r in records}):
        write_manifest(root/'manifests'/f'{split}.json',[r for r in records if r['split']==split])
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','development','holdout']);args=p.parse_args()
    rows=prepare(stage=args.stage);print(f'{args.stage}: {len(rows)} immutable scene definitions')
