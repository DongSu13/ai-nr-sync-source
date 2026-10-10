"""Local supervised campaign: bounded MATLAB workers, fail-stop phase gates."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from matlab_launcher import resolve_matlab_path
from nr_v3_campaign import ROOT, DATA, prepare, receiver_hash

MATLAB = None


def partitions(count, workers):
    if count<0 or workers<1:raise ValueError('invalid partition')
    step=max(1,math.ceil(count/workers))
    return [(first,min(count,first+step-1)) for first in range(1,count+1,step)]


def state(phase, **details):
    value=dict(phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**details)
    temporary=DATA/'pipeline_status.tmp';temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(DATA/'pipeline_status.json');print(json.dumps(value),flush=True)


def failed_scene_context(records,root=DATA):
    """Locate the first incomplete/invalid record in the sequential worker range."""
    for record in records:
        path=Path(root)/'scenes'/record['split']/record['cell']/f"{record['seed']}.json"
        try:
            scene=json.loads(path.read_text())
            if any(scene[k]!=record[k] for k in ('seed','config_hash','receiver_hash')) or len(scene['candidates'])!=12:
                return record
        except (OSError,ValueError,KeyError,TypeError):return record
    return None


def matlab_worker(job):
    split,first,last=job;log=DATA/'logs'/f'{split}_{first}_{last}.log'
    expression=f"addpath('src/matlab');run_nr_v3('{split}',{first},{last})"
    matlab=MATLAB if MATLAB is not None else resolve_matlab_path()
    argv=[str(matlab),'-wait','-batch',expression,'-logfile',str(log)]
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (DATA/'logs'/f'{split}_{first}_{last}_console.log').open('a',encoding='utf-8') as stream:
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
        print(f'WORKER {split} {first}..{last} PID={proc.pid}',flush=True)
        code=proc.wait()
    if code:
        records=json.loads((DATA/'manifests'/f'{split}.json').read_text())[first-1:last]
        failed=failed_scene_context(records)
        from nr_v3_experiment import write_new_json
        error=dict(exit_code=code,worker_range=[first,last],scene=failed,log=str(log),
                   scene_identification='first incomplete/invalid sequential record; inferred from outputs',
                   error_excerpt=log.read_text(errors='replace')[-8000:] if log.exists() else 'MATLAB logfile absent')
        write_new_json(DATA/'errors'/f'{split}_{first}_{last}_{time.time_ns()}.json',error)
        raise RuntimeError(f'MATLAB failed ({code}): {log}')
    return f'{split}:{first}-{last}'


def generate(splits,workers):
    from nr_v3_experiment import split_scenes
    jobs=[]
    for split in splits:
        manifest=json.loads((DATA/'manifests'/f'{split}.json').read_text())
        jobs += [(split,a,b) for a,b in partitions(len(manifest),workers)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(matlab_worker,job) for job in jobs]
        try:
            for future in as_completed(futures):print('WORKER DONE',future.result(),flush=True)
        except BaseException:
            for future in futures:future.cancel()
            # Already-running owned workers finish atomically; queued jobs are cancelled.
            raise
    for split in splits:split_scenes(split)


def run(stage,workers):
    DATA.mkdir(parents=True,exist_ok=True);(DATA/'logs').mkdir(exist_ok=True)
    lock=DATA/'pipeline.lock'
    with lock.open('x',encoding='utf-8') as stream:json.dump(dict(pid=os.getpid(),stage=stage),stream)
    try:
        state('starting',stage=stage,receiver_hash=receiver_hash(),workers=workers)
        if stage in ('development','all'):
            prepare(stage='development');state('generating_development')
            generate(['train','validation','development'],workers)
            state('development_complete')
        if stage in ('final','all'):
            from nr_v3_experiment import train,freeze,evaluate_suite,audit,check_hashes
            if not (DATA/'freeze.json').exists():
                state('training');train();state('freezing');freeze()
            frozen=json.loads((DATA/'freeze.json').read_text());check_hashes(frozen['source_hashes']);check_hashes(frozen['model_hashes'])
            prepare(stage='holdout');state('generating_holdouts')
            generate(['test','sweep','stress','noise'],workers)
            state('evaluating')
            for split in ['test','sweep','stress','noise']:
                if not (DATA/f'{split}_evaluation.json').exists():evaluate_suite(split)
            if not (DATA/'integrity_audit.json').exists():audit()
            state('reporting')
            subprocess.run([sys.executable,str(ROOT/'src/python/report_nr_v3.py')],cwd=ROOT,check=True)
            state('complete',report=str(DATA/'expanded_report.md'))
    except BaseException as exc:
        state('failed',error=repr(exc));raise
    finally:
        # Only this process's explicitly acquired marker, never scene/model data.
        lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','final','all']);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.workers<=3:p.error('workers must be 1..3 on this local machine')
    run(a.stage,a.workers)
