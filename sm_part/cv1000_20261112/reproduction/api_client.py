"""Minimal OpenAI client. Keys stay in memory and never enter artifacts/logs."""
import argparse
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://api.openai.com/v1/'

def read_key(path):
    raw = Path(path).read_text(encoding='utf-8-sig').strip()
    found = re.findall(r'sk-[A-Za-z0-9_-]+', raw)
    if len(found) != 1:
        raise RuntimeError('Key file must contain exactly one OpenAI API key; contents suppressed.')
    return found[0]

def request(key, route, body=None):
    req = urllib.request.Request(BASE_URL + route,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            started = time.perf_counter()
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.load(response)
                return {'response': data, 'request_id': response.headers.get('x-request-id'),
                        'latency_seconds': time.perf_counter() - started}
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read()).get('error', {})
            except Exception:
                err = {}
            code = err.get('code')
            if (exc.code == 429 and code != 'insufficient_quota' or exc.code >= 500) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            # Do not echo provider messages, which may contain account/key fragments.
            raise RuntimeError(f'OpenAI HTTP {exc.code}; code={code}; type={err.get("type")}') from None
        except urllib.error.URLError:
            raise RuntimeError('OpenAI network connection failed; request details suppressed.') from None

def probe(key_path):
    key = read_key(key_path)
    response = request(key, 'models')['response']
    names = sorted(m['id'] for m in response['data'] if m['id'].startswith('gpt-3.5'))
    result = {'accessible_gpt35_models': names, 'paid_generation_performed': False}
    output = ROOT / 'results' / 'model_access.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key-file', required=True)
    args = parser.parse_args()
    try:
        probe(args.key_file)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)
