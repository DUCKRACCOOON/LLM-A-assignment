"""Resumable BF16 Llama 3 reproduction. No OpenAI API requests.

The author did not pin Transformers. This run pins 4.44.2 and records the
actual GPU runtime. The original raw Llama prompts (including special tokens)
are passed through the pipeline without applying an additional chat template.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import runpy
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from core import ROOT, VENDOR, World, load_cases, parse_path, path_length, search, validate

MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
KINDS = ('standard', 'cot', 'repe')
OUT = ROOT / 'results' / 'llama'
PROMPTS = runpy.run_path(str(VENDOR / 'llmastar/pather/llm_a_star/prompt.py'))['llama_prompt']
VENDOR_COMMIT = '9414ed894d774f71c36e8c0afd0e20e6033b95b1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def status(phase, **kw):
    record = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), **kw}
    save(OUT / 'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def cases_for(stage):
    if stage == 'main':
        return load_cases()
    return [dict(id=f'dev-{i:02d}', map_id=-1-i, pair_id=0, range_x=[0, 51], range_y=[0, 31],
                 start=[3+i, 4], goal=[43-i, 25], horizontal_barriers=[[12, 0, 24+i]],
                 vertical_barriers=[[33, 17, 30]]) for i in range(10)]


def tasks_for(stage):
    return [(q, k) for q in cases_for(stage) for k in KINDS]


def protocol():
    return {'model_id': MODEL, 'vendor_commit': VENDOR_COMMIT,
            'dataset_sha256': file_hash(VENDOR / 'dataset/environment_50_30.json'),
            'llama_prompts_sha256': digest(PROMPTS),
            'core_sha256': file_hash(ROOT / 'reproduction/core.py'),
            'runner_sha256': file_hash(Path(__file__)),
            'dtype': 'bfloat16', 'do_sample': False, 'temperature': None, 'top_p': None,
            'batch_size': 1, 'nominal_max_new_tokens': 8000,
            'context_policy': 'min(8000, model_max_position_embeddings - input_tokens)',
            'prompt_serialization': 'author raw string; pipeline default add_special_tokens',
            'attention_implementation': 'sdpa', 'seed': 240702511,
            'historical_transformers_version_unknown': True}


def preflight(device):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run inside the allocated Sirius GPU session.')
    torch.cuda.set_device(device)
    devices = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
            devices.append({'index': i, 'name': torch.cuda.get_device_name(i),
                            'free_bytes': free, 'total_bytes': total,
                            'bf16_supported': torch.cuda.is_bf16_supported()})
    if not devices[device]['bf16_supported']:
        raise RuntimeError('The selected GPU runtime does not support BF16.')
    if devices[device]['free_bytes'] < 18 * 1024**3:
        raise RuntimeError('Less than 18 GiB GPU memory is free. Check other workloads before loading.')
    versions = {}
    for name in ('torch', 'transformers', 'tokenizers', 'accelerate', 'huggingface-hub', 'safetensors', 'shapely'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if versions['transformers'] != '4.44.2':
        raise RuntimeError('Install the pinned requirements-llama.txt environment first.')
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'versions': versions, 'cuda': torch.version.cuda, 'devices': devices,
              'disk_free_bytes': shutil.disk_usage(ROOT).free,
              'project_path': str(ROOT), 'selected_device': device}
    save(OUT / 'preflight.json', result)
    return result


def download(token_file=None, prompt_token=False):
    manifest_path = OUT / 'model_manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        model_path = ROOT / manifest['snapshot_relative_path']
        if manifest['model_id'] != MODEL:
            raise RuntimeError('Cached model identity differs from the paper model.')
        for name, entry in manifest['files'].items():
            p = model_path / name
            if not p.is_file() or p.stat().st_size != entry['bytes']:
                raise RuntimeError(f'Cached model file is missing or incomplete: {name}')
        return model_path, manifest
    from huggingface_hub import HfApi, get_token, snapshot_download
    token = Path(token_file).read_text(encoding='utf-8').strip() if token_file else get_token()
    if not token and prompt_token:
        from getpass import getpass
        token = getpass('Hugging Face read token (hidden; not saved): ').strip()
    if not token:
        raise RuntimeError('HF authentication required. Use an approved account via --hf-token-file or --prompt-token.')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('At least 20 GiB free persistent disk space is required for initial download.')
    status('downloading', model_id=MODEL)
    info = HfApi().model_info(MODEL, token=token)
    revision = info.sha
    snapshot = Path(snapshot_download(
        MODEL, revision=revision, token=token, cache_dir=str(ROOT / 'models/hub'),
        allow_patterns=['*.json', 'model*.safetensors', 'tokenizer.model', 'LICENSE', 'USE_POLICY.md', 'README.md'],
        ignore_patterns=['original/*'], max_workers=4))
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
             for p in sorted(snapshot.iterdir()) if p.is_file()}
    if not any(name.endswith('.safetensors') for name in files):
        raise RuntimeError('No Safetensors model weights were downloaded.')
    manifest = {'model_id': MODEL, 'revision': revision,
                'snapshot_relative_path': snapshot.relative_to(ROOT).as_posix(), 'files': files}
    save(manifest_path, manifest)
    return snapshot, manifest


def token_budget(input_tokens, context_length):
    if input_tokens >= context_length:
        raise RuntimeError(f'Prompt has {input_tokens} tokens, exceeding the {context_length} token context.')
    return min(8000, context_length - input_tokens)


class Generator:
    def __init__(self, model_path, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
        self.torch = torch
        self.device = device
        set_seed(240702511)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, device_map={'': device},
            low_cpu_mem_usage=True, attn_implementation='sdpa', local_files_only=True)
        model.eval()
        self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
        self.context_length = model.config.max_position_embeddings
        self.terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        if None in self.terminators or self.terminators[1] == tokenizer.unk_token_id:
            raise RuntimeError('Expected Llama 3 EOS/EOT tokens were not found.')

    def inspect(self, prompt):
        ids = self.pipe.preprocess(prompt)['input_ids'][0].tolist()
        return {'input_ids': ids, 'input_tokens': len(ids),
                'input_ids_sha256': digest(ids), 'max_new_tokens': token_budget(len(ids), self.context_length),
                'bos_token_count': ids.count(self.pipe.tokenizer.bos_token_id)}

    def generate(self, prompt, spec):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        # Tensor output preserves exact output IDs without making a second request.
        outputs = self.pipe(prompt, max_new_tokens=spec['max_new_tokens'],
                            eos_token_id=self.terminators, do_sample=False,
                            temperature=None, top_p=None,
                            pad_token_id=self.pipe.tokenizer.eos_token_id, return_tensors=True)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0
        sequence = outputs[0]['generated_token_ids']
        if hasattr(sequence, 'tolist'):
            sequence = sequence.tolist()
        if sequence[:spec['input_tokens']] != spec['input_ids']:
            raise RuntimeError('Pipeline tokenization changed between preflight and generation.')
        generated = sequence[spec['input_tokens']:]
        # Match pipeline NEW_TEXT postprocessing (decoded full text minus decoded prompt).
        tokenizer = self.pipe.tokenizer
        full = tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        prefix = tokenizer.decode(spec['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        content = full[len(prefix):]
        return {'content': content, 'generated_token_ids': generated, 'completion_tokens': len(generated),
                'finish_reason': 'stop' if generated and generated[-1] in self.terminators else 'length',
                'elapsed_seconds': elapsed,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device)}


def run_identity(manifest, environment):
    return {'protocol': protocol(), 'model_revision': manifest['revision'],
            'model_files': manifest['files'], 'versions': environment['versions'],
            'cuda': environment['cuda'], 'python': environment['python'],
            'gpu_name': environment['devices'][environment['selected_device']]['name']}


def check_record(record, run_id, q, kind):
    expected = digest({'case': q, 'prompt': PROMPTS[kind].format(**q), 'run_id': run_id})
    if (record.get('run_id') != run_id or record.get('request_sha256') != expected
            or record.get('case_id') != q['id'] or record.get('prompt_kind') != kind):
        raise RuntimeError(f'Cached request mismatch: {q["id"]}/{kind}')
    if record.get('result_sha256') != digest(record.get('result')):
        raise RuntimeError(f'Cached response checksum mismatch: {q["id"]}/{kind}')


def collect(stage, generator, identity):
    out = OUT / stage
    run_id = digest(identity)
    stage_manifest = {'run_id': run_id, 'identity': identity, 'cases': cases_for(stage), 'kinds': list(KINDS)}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and read(manifest_path) != stage_manifest:
        raise RuntimeError('The saved run environment differs; refusing to mix conditions.')
    save(manifest_path, stage_manifest)
    tasks = tasks_for(stage)
    # Inspect every input before the first paid/expensive generation; never truncate prompts.
    specs = {(q['id'], k): generator.inspect(PROMPTS[k].format(**q)) for q, k in tasks}
    save(out / 'token_audit.json', {
        'context_length': generator.context_length,
        'max_input_tokens': max(v['input_tokens'] for v in specs.values()),
        'min_output_budget': min(v['max_new_tokens'] for v in specs.values()),
        'bos_token_counts': sorted({v['bos_token_count'] for v in specs.values()}),
        'note': 'Author raw prompt and pipeline special-token behavior preserved, including any duplicate BOS.'})
    for index, (q, kind) in enumerate(tasks):
        path = out / 'raw' / f'{q["id"]}-{kind}.json'
        if path.exists():
            check_record(read(path), run_id, q, kind)
        else:
            prompt = PROMPTS[kind].format(**q)
            spec = specs[q['id'], kind]
            result = generator.generate(prompt, spec)
            record = {'case_id': q['id'], 'prompt_kind': kind, 'run_id': run_id,
                      'request_sha256': digest({'case': q, 'prompt': prompt, 'run_id': run_id}),
                      'created_utc': datetime.now(timezone.utc).isoformat(), 'prompt': prompt,
                      'generation': {k: v for k, v in spec.items() if k != 'input_ids'},
                      'result': result, 'result_sha256': digest(result)}
            save(path, record)
        if (index + 1) % 10 == 0 or index == 0:
            status(f'{stage}_generation', completed=index + 1, total=len(tasks))


def evaluate(stage):
    out = OUT / stage
    manifest = read(out / 'manifest.json')
    if manifest['cases'] != cases_for(stage) or manifest['identity']['protocol'] != protocol():
        raise RuntimeError('Evaluation inputs or source differ from the recorded generation protocol.')
    expected = {f'{q["id"]}-{k}.json' for q, k in tasks_for(stage)}
    actual = {p.name for p in (out / 'raw').glob('*.json')}
    if actual != expected:
        raise RuntimeError(f'Raw responses incomplete: expected {len(expected)}, found {len(actual)}.')
    rows, baselines, worlds = [], [], {}
    for index, q in enumerate(cases_for(stage)):
        if q['map_id'] not in worlds:
            worlds[q['map_id']] = World(q)
        world = worlds[q['map_id']]
        records = {k: read(out / 'raw' / f'{q["id"]}-{k}.json') for k in KINDS}
        for kind, record in records.items():
            check_record(record, manifest['run_id'], q, kind)
        signature = digest({'case': q, 'raw': records, 'core': protocol()['core_sha256']})
        p = out / 'evaluation' / f'{q["id"]}.json'
        if p.exists() and read(p).get('signature') == signature:
            saved = read(p)
        else:
            baseline = {'case_id': q['id'], 'method': 'astar', 'prompt': None, **search(q, 'astar', world=world)}
            results = []
            for kind, record in records.items():
                path, error = parse_path(record['result']['content'])
                valid, failure = validate(q, path, world)
                results.append({'case_id': q['id'], 'method': 'llm_only', 'prompt': kind,
                                'operation': None, 'storage': None, 'length': path_length(path) if path else None,
                                'valid': valid, 'failure': error or failure, 'parse_error': error, 'path': path})
                results.append({'case_id': q['id'], 'method': 'llm_astar', 'prompt': kind,
                                'parse_error': error, **search(q, 'llm_astar', path, world)})
            saved = {'signature': signature, 'case': q, 'baseline': baseline, 'rows': results}
            save(p, saved)
        rows.extend(saved['rows'])
        baselines.append(saved['baseline'])
        if (index + 1) % 50 == 0:
            status(f'{stage}_evaluation', completed=index + 1, total=len(cases_for(stage)))
    save(out / 'rows.json', rows)
    save(out / 'baselines.json', baselines)
    return rows


def verify(stage):
    out = OUT / stage
    manifest = read(out / 'manifest.json')
    if manifest['cases'] != cases_for(stage) or manifest['identity']['protocol'] != protocol():
        raise RuntimeError('Verification protocol mismatch.')
    expected_files = {f'{q["id"]}-{k}.json' for q, k in tasks_for(stage)}
    if {p.name for p in (out / 'raw').glob('*.json')} != expected_files:
        raise RuntimeError('Missing or unexpected raw response files.')
    finish, timings, peaks = collections.Counter(), collections.defaultdict(list), []
    for q, k in tasks_for(stage):
        record = read(out / 'raw' / f'{q["id"]}-{k}.json')
        check_record(record, manifest['run_id'], q, k)
        r = record['result']
        if r['completion_tokens'] != len(r['generated_token_ids']):
            raise RuntimeError('Output token count mismatch.')
        finish[r['finish_reason']] += 1
        timings[k].append(r['elapsed_seconds'])
        peaks.append(r['peak_reserved_bytes'])
    rows = read(out / 'rows.json')
    expected_rows = {(q['id'], m, k) for q, k in tasks_for(stage) for m in ('llm_only', 'llm_astar')}
    keys = [(r['case_id'], r['method'], r['prompt']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_rows:
        raise RuntimeError('Evaluation rows contain missing or duplicate conditions.')
    result = {'complete': True, 'stage': stage, 'case_count': len(cases_for(stage)),
              'responses': len(expected_files), 'result_rows': len(rows), 'run_id': manifest['run_id'],
              'finish_reasons': dict(finish), 'peak_reserved_bytes': max(peaks),
              'mean_seconds_by_prompt': {k: statistics.mean(v) for k, v in timings.items()},
              'estimated_main_generation_hours_single_gpu': 1000 * sum(statistics.mean(v) for v in timings.values()) / 3600}
    save(out / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'download', 'collect', 'evaluate', 'verify', 'report', 'all'])
    parser.add_argument('--stage', choices=['pilot', 'main'], default='pilot')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hf-token-file')
    parser.add_argument('--prompt-token', action='store_true')
    args = parser.parse_args()
    if args.action in ('all', 'collect') and os.name == 'posix':
        import fcntl
        OUT.mkdir(parents=True, exist_ok=True)
        lock = (OUT / 'runner.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Llama collector is already running in this project.')
    if args.action == 'evaluate':
        evaluate(args.stage)
    elif args.action == 'verify':
        verify(args.stage)
    elif args.action == 'report':
        from llama_report import main as report
        report()
    elif args.action == 'download':
        download(args.hf_token_file, args.prompt_token)
    else:
        environment = preflight(args.device)
        if args.action == 'preflight':
            print(json.dumps(environment, indent=2))
            return
        model_path, manifest = download(args.hf_token_file, args.prompt_token)
        status('loading_model', device=args.device)
        generator = Generator(model_path, args.device)
        identity = run_identity(manifest, environment)
        stages = ('pilot', 'main') if args.action == 'all' else (args.stage,)
        for stage in stages:
            collect(stage, generator, identity)
            if args.action == 'all':
                evaluate(stage)
                verify(stage)
        if args.action == 'all':
            from llama_report import main as report
            report()
            status('complete', responses=3000, result_rows=6000)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        message = re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', str(exc))
        status('failed', error_type=type(exc).__name__, error=message)
        raise SystemExit(1)
