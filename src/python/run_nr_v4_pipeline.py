"""Single-worker local supervisor; exclusive ownership and immutable receipts."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from matlab_launcher import resolve_matlab_path
from nr_v4_campaign import ROOT,DATA,prepare,receiver_hash,write_immutable

MATLAB = None


def hash_files(paths):
    return {str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def check_hashes(snapshot):
    for p,sha in snapshot.items():
        if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'file changed: {p}')


@contextmanager
def campaign_lock(root=DATA):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'pipeline.lock'
    with path.open('x',encoding='utf-8') as stream:json.dump(dict(pid=os.getpid(),workers=1),stream)
    try:yield
    finally:path.unlink()


def state(phase,**details):
    value=dict(phase=phase,pid=os.getpid(),workers=1,updated_utc=datetime.now(timezone.utc).isoformat(),**details)
    tmp=DATA/'pipeline_status.tmp';tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(DATA/'pipeline_status.json')
    print(json.dumps(value),flush=True)


def matlab_job(expression,name):
    logs=DATA/'logs';logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    matlab=MATLAB if MATLAB is not None else resolve_matlab_path()
    command=[str(matlab),'-nojvm','-singleCompThread','-wait','-batch',"addpath('src/matlab');"+expression,
             '-logfile',str(logs/f'{name}.log')]
    with (logs/f'{name}_console.log').open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        print(f'ONE MATLAB WORKER pid={process.pid} {name}',flush=True)
        code=process.wait()
    if code:raise RuntimeError(f'MATLAB exit {code}; inspect {logs/name}')


def collection_sources():
    return sorted((ROOT/'src/matlab').glob('nr_v4_*.m'))+[
        ROOT/'src/matlab/nr_v3_grid.m',ROOT/'src/matlab/run_nr_v4.m',
        ROOT/'src/python/nr_v4_campaign.py',ROOT/'src/python/nr_v4_audit.py']


def development():
    from nr_v4_audit import load_collection,seq
    sources=hash_files(collection_sources())
    receipt=dict(receiver_hash=receiver_hash(),source_hashes=sources)
    write_immutable(DATA/'collection_recipe.json',receipt)
    prepare(stage='development')
    for split,count in (('train',1200),('development',200)):
        check_hashes(sources);state('collecting',split=split,expected_scenes=count)
        matlab_job(f"run_nr_v4('{split}',1,{count},'collect')",f'collect_{split}')
        check_hashes(sources)
        samples=load_collection(split)
        summary=dict(split=split,scenes=len(samples),rank_rows=sum(len(s['rank_y']) for s in samples),
                     positive_rows=sum(sum(s['rank_y']) for s in samples),branches=sum(len(seq(s['branches'])) for s in samples),
                     total_seconds=sum(s['elapsed_seconds'] for s in samples),receiver_hash=receiver_hash(),
                     data_hashes=hash_files(sorted((DATA/'collections'/split).glob('*.json'))))
        write_immutable(DATA/f'{split}_collection_audit.json',summary)
        print('COLLECTION AUDIT PASS',split,len(samples),flush=True)
    state('development_complete',next_stage='validation selection, model freezing and new holdout evaluation remain')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','final','all']);args=p.parse_args()
    with campaign_lock():
        try:
            if args.stage in ('development','all'):development()
            if args.stage in ('final','all'):final_stage()
        except BaseException as exc:state('failed',error=repr(exc));raise


def final_stage():
    from nr_v4_train import select_and_freeze,evaluate_job
    from nr_v4_results import audit_all,summarize
    from report_nr_v4 import main as report
    for split in ('train','development'):
        receipt=json.loads((DATA/f'{split}_collection_audit.json').read_text());check_hashes(receipt['data_hashes'])
    check_hashes(json.loads((DATA/'collection_recipe.json').read_text())['source_hashes'])
    state('training_and_selection');lock=select_and_freeze();prepare(stage='holdout')
    for split in ('test','stress','noise'):
        check_hashes(lock['source_hashes']);check_hashes(lock['model_hashes']);state('evaluating_holdout',split=split)
        evaluate_job(lock['final_job'],split)
    state('auditing');audit_all();summarize();state('reporting');report()
    state('complete',report=str(DATA/'closed_loop_report.md'))


if __name__=='__main__':main()
