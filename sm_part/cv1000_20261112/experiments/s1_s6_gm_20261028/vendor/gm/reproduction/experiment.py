"""Run: python -m reproduction.experiment --help"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import runpy
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from .api import completion, load_key, ENDPOINT
from .core import Geometry, search, llm_only

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'source_code/dataset/environment_50_30.json'
PROMPT_FILE = ROOT / 'source_code/llmastar/pather/llm_a_star/prompt.py'
MODELS = ['gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125']
PROMPTS = ['standard', 'cot', 'repe']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    result = []
    for mi, data in enumerate(json.loads(DATASET.read_text(encoding='utf-8'))):
        for pi, pair in enumerate(data['start_goal']):
            query = {k: data[k] for k in ('range_x', 'range_y', 'horizontal_barriers', 'vertical_barriers')}
            query.update(start=pair[0], goal=pair[1])
            result.append(dict(case_id=f'm{mi:03d}_p{pi:02d}', map_index=mi, pair_index=pi,
                               query=query, natural_query=pair[2] if len(pair)>2 else None))
    return result


def records(path):
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError as exc:
                raise RuntimeError(f'Corrupt checkpoint {path.name}:{number}; preserve and repair explicitly') from exc
    return result


def append(file, value):
    file.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    file.flush()
    os.fsync(file.fileno())


def manifest(out, settings):
    value = dict(settings=settings, dataset_sha256=sha(DATASET), prompt_sha256=sha(PROMPT_FILE),
                 code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'reproduction').glob('*.py'))},
                 python=platform.python_version(), platform=platform.platform(),
                 endpoint=ENDPOINT, utc=datetime.now(timezone.utc).isoformat(),
                 upstream_commit='9414ed894d774f71c36e8c0afd0e20e6033b95b1',
                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    out.mkdir(parents=True, exist_ok=True)
    filename = out / ('manifest_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
    filename.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return filename.name


def selected_cases(limit):
    all_cases = cases()
    # Pilot: three pairs from each of ten evenly spaced maps (not first 30).
    if limit == 30:
        return [c for c in all_cases if c['map_index'] % 10 == 0 and c['pair_index'] < 3]
    return all_cases[:limit] if limit else all_cases


def generate(args):
    out = Path(args.output)
    selected = selected_cases(args.limit)
    prompts = runpy.run_path(str(PROMPT_FILE))['gpt_prompt']
    key = load_key()
    manifest_name = manifest(out, vars(args))
    previous = records(out/'generations.jsonl')
    cached = {r['request_sha256'] for r in previous}
    jobs = []
    for case in selected:
        for model in args.models:
            for style in PROMPTS:
                q = case['query']
                prompt = prompts[style].format(start=q['start'], goal=q['goal'],
                                              horizontal_barriers=q['horizontal_barriers'],
                                              vertical_barriers=q['vertical_barriers'])
                body = dict(model=model, messages=[dict(role='user', content=prompt)],
                            temperature=0, max_tokens=args.max_tokens, store=False)
                fingerprint = hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
                if fingerprint in cached:
                    continue
                jobs.append(dict(case_id=case['case_id'], map_index=case['map_index'],
                                 pair_index=case['pair_index'], model=model, prompt_style=style,
                                 request=body, request_sha256=fingerprint, manifest=manifest_name))
    print(json.dumps(dict(pending=len(jobs), cached=len(previous), workers=args.workers)), flush=True)
    started, done, failures = time.perf_counter(), 0, 0

    def execute(job):
        return dict(**job, utc=datetime.now(timezone.utc).isoformat(), **completion(job['request'],key))

    # Bounded in-flight work allows fatal authentication/quota errors to stop
    # promptly, without scheduling thousands of doomed requests.
    with (out/'generations.jsonl').open('a',encoding='utf-8') as file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(jobs)
            active = {pool.submit(execute, job) for job in [next(iterator, None) for _ in range(args.workers)] if job}
            stop = False
            while active:
                completed, active = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    append(file, result)
                    done += 1
                    failures += result['status'] != 'ok'
                    if result['status'] != 'ok':
                        last = result['attempts'][-1]
                        if last.get('http_status') in (401,403,404) or last.get('error_code') == 'insufficient_quota':
                            stop = True
                    if done % 30 == 0 or result['status'] != 'ok' or done == len(jobs):
                        print(json.dumps(dict(completed=done, planned=len(jobs), failures=failures,
                                              elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
                    job = None if stop else next(iterator, None)
                    if job:
                        active.add(pool.submit(execute,job))
            if stop:
                raise SystemExit('Stopped on nonretryable account/model error; checkpoints retained.')


def evaluate(args):
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    manifest_name = manifest(out, vars(args))
    all_cases = {c['case_id']:c for c in selected_cases(args.limit)}
    geometries = {}
    existing = records(out/'evaluations.jsonl')
    cached = {r['evaluation_id'] for r in existing}
    count = 0
    with (out/'evaluations.jsonl').open('a',encoding='utf-8') as file:
        for case in all_cases.values():
            mi = case['map_index']
            geo = geometries.setdefault(mi, Geometry(case['query']))
            for method in ('astar','dwa','goal_only'):
                eid = case['case_id'] + ':' + method
                if eid not in cached:
                    value = dict(case_id=case['case_id'], map_index=mi, method=method,
                                 model=None, prompt_style=None, evaluation_id=eid,manifest=manifest_name,
                                 **search(case['query'],geo,method))
                    append(file,value)
                    count += 1
            if count and count % 150 == 0:
                print(json.dumps(dict(evaluations_added=count,case_id=case['case_id'])),flush=True)
        for gen in records(out/'generations.jsonl'):
            if gen['case_id'] not in all_cases:
                continue
            case = all_cases[gen['case_id']]
            geo = geometries[case['map_index']]
            for method in ('llm_only','llm_astar'):
                eid = gen['request_sha256'] + ':' + method
                if eid in cached:
                    continue
                if gen['status'] != 'ok':
                    metrics = dict(valid=False,error='api_error',length=None,operation=None,storage=None,path=[])
                else:
                    content = gen['response']['choices'][0]['message'].get('content') or ''
                    metrics = (llm_only(case['query'],geo,content) if method == 'llm_only'
                               else search(case['query'],geo,method,text=content))
                append(file,dict(case_id=case['case_id'],map_index=case['map_index'],method=method,
                                 model=gen['model'],prompt_style=gen['prompt_style'],evaluation_id=eid,
                                 request_sha256=gen['request_sha256'],manifest=manifest_name,**metrics))
                cached.add(eid)
                count += 1
                if count % 300 == 0:
                    print(json.dumps(dict(evaluations_added=count)),flush=True)
    print(json.dumps(dict(evaluations_added=count, total=len(existing)+count)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['generate','evaluate'])
    parser.add_argument('--output',default='results/main')
    parser.add_argument('--limit',type=int,default=0,help='0=all 1000; 30=prespecified pilot')
    parser.add_argument('--models',nargs='+',default=MODELS)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--max-tokens',type=int,default=1000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    (generate if args.action == 'generate' else evaluate)(args)


if __name__ == '__main__':
    main()
