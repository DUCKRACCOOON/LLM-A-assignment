"""Build a credential-free, explicit allowlist of files for the Sirius server."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from report import PAPER
    summary = json.loads((ROOT / 'results/report/summary.json').read_text(encoding='utf-8'))['main']
    table = []
    for item in summary:
        row = dict(item, base_model='GPT-3.5-Turbo' if item['prompt'] else '—')
        for metric, value in zip(('operation', 'storage', 'length', 'valid'), PAPER[item['method'], item['prompt']]):
            row['paper_' + metric + '_ratio'] = value
        table.append(row)
    rows = json.loads((ROOT / 'results/main/rows.json').read_text(encoding='utf-8'))
    baselines = [{key: r[key] for key in ('case_id', 'operation', 'storage', 'length', 'valid')}
                 for r in rows if r['method'] == 'astar']
    ref = ROOT / 'reproduction/llama_reference.json'
    ref.write_text(json.dumps({'gpt_table1': table, 'baselines': baselines}, ensure_ascii=False, indent=2), encoding='utf-8')
    files = ['requirements-llama.txt', 'reproduction/core.py', 'reproduction/llama_run.py',
             'reproduction/llama_report.py', 'reproduction/llama_reference.json', 'reproduction/test_llama.py',
             'reproduction/server_smoke.py',
             'reproduction/server_setup.sh', 'reproduction/server_start.sh',
             'vendor/llm-astar/dataset/environment_50_30.json',
             'vendor/llm-astar/llmastar/pather/llm_a_star/prompt.py', 'vendor/llm-astar/LICENSE']
    out = ROOT / 'dist'
    out.mkdir(exist_ok=True)
    archive = out / 'llama-reproduction.zip'
    hashes = {}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in files:
            data = (ROOT / name).read_bytes()
            if name.endswith('.sh'):
                data = data.replace(b'\r\n', b'\n')
            hashes[name] = hashlib.sha256(data).hexdigest()
            z.writestr('llama-reproduction/' + name, data)
        z.writestr('llama-reproduction/bundle_manifest.json', json.dumps(hashes, indent=2))
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'files': len(files), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
