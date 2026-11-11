"""Final completeness/provenance audit; no API calls and no key access."""
import collections
import hashlib
import json
from pathlib import Path
from core import ROOT, PROMPTS
from run import digest, save

def verify():
    checks={}
    for stage,expected_cases,expected_responses,expected_rows in [
            ('pilot',10,30,80),('main',1000,3000,8000),('scaling',100,100,300)]:
        folder=ROOT/'results'/stage
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        responses=[json.loads(p.read_text(encoding='utf-8')) for p in (folder/'raw').glob('*.json')]
        rows=json.loads((folder/'rows.json').read_text(encoding='utf-8'))
        assert manifest['case_count']==expected_cases,(stage,'case count')
        assert len(responses)==expected_responses,(stage,'response count')
        assert len(rows)==expected_rows,(stage,'row count')
        assert manifest['prompts_sha256']==digest(PROMPTS),(stage,'prompt drift')
        assert len({(r['case_id'],r['prompt_kind']) for r in responses})==expected_responses
        assert len({(r['case_id'],r['method'],r['prompt']) for r in rows})==expected_rows
        ids={q['id'] for q in manifest['cases']}
        assert {r['case_id'] for r in rows}==ids
        for r in responses:
            assert r['payload_sha256']==digest(r['payload'])
            assert r['payload']['model']=='gpt-3.5-turbo-0125'
            assert r['response']['model']=='gpt-3.5-turbo-0125'
            assert r['response']['choices'][0]['finish_reason'] in ('stop','length','content_filter')
        expected_per_case=3 if stage=='scaling' else 8
        assert set(collections.Counter(r['case_id'] for r in rows).values())=={expected_per_case}
        checks[stage]={'cases':expected_cases,'responses':len(responses),'result_rows':len(rows),
                       'finishes':dict(collections.Counter(r['response']['choices'][0]['finish_reason'] for r in responses)),
                       'valid_counts':dict(collections.Counter(r['method']+'/'+str(r['prompt']) for r in rows if r['valid']))}
    sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (ROOT/'reproduction').glob('*.py')}
    result={'complete':True,'checks':checks,'source_sha256':sources}
    save(ROOT/'results/verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': verify()
