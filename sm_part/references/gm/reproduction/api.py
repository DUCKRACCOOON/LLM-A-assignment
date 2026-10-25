"""Small auditable OpenAI Chat Completions adapter; no SDK dependency."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = 'https://api.openai.com/v1/chat/completions'


def load_key():
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        env = Path(__file__).resolve().parents[1] / '.env'
        if env.exists():
            for line in env.read_text(encoding='utf-8-sig').splitlines():
                if line.startswith('OPENAI_API_KEY='):
                    key = line.split('=', 1)[1].strip().strip('\"\'')
    if not key:
        raise RuntimeError('Set OPENAI_API_KEY or an ignored local .env file.')
    return key


def completion(body, key=None, attempts=4):
    key = key or load_key()
    started = time.perf_counter()
    history = []
    for attempt in range(attempts):
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                        headers={'Authorization': 'Bearer '+key,
                                                 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
                headers = {k:v for k,v in response.headers.items()
                           if k.lower().startswith('x-ratelimit') or k.lower() == 'x-request-id'}
            return dict(status='ok', response=result, headers=headers, attempts=history,
                        seconds=time.perf_counter()-started)
        except urllib.error.HTTPError as exc:
            # Do not log raw error messages, request headers, or credentials.
            try:
                error = json.loads(exc.read()).get('error', {})
            except (ValueError, AttributeError):
                error = {}
            history.append(dict(http_status=exc.code, error_type=error.get('type'),
                                error_code=error.get('code')))
            if exc.code not in (408, 429, 500, 502, 503, 504) or error.get('code') == 'insufficient_quota':
                break
            if attempt + 1 < attempts:
                time.sleep(min(30, 2 ** (attempt + 1)))
        except (urllib.error.URLError, TimeoutError, OSError):
            history.append(dict(error_type='transport_error'))
            if attempt + 1 < attempts:
                time.sleep(min(30, 2 ** (attempt + 1)))
    return dict(status='api_error', attempts=history, seconds=time.perf_counter()-started)
