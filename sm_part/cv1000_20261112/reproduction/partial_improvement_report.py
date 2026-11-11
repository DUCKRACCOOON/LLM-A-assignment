"""Audit completed artifacts and export an explicitly incomplete report."""
import argparse
import hashlib
import html
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'reproduction'))
    from core import validate, path_length
    from run import cases_for, digest, payload, save
    from improve_core import correction_payload
    from improve_run import protocol_signature, text_of
    from improve_report import summary, write_csv, table, LABEL, PROMPT

    source = root / 'results/improvements'
    out = root / 'results/improvements_partial'
    out.mkdir(parents=True, exist_ok=True)
    status = json.loads((source / 'status.json').read_text(encoding='utf-8'))
    assert status['phase'] == 'failed' and 'insufficient_quota' in status['error']
    frozen = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    assert frozen['signature'] == protocol_signature()
    allrows = {}
    checks = {}
    sig = protocol_signature()
    for stage in ('pilot', 'main', 'scaling'):
        qs = {q['id']: q for q in cases_for(stage)}
        rows = []
        seen = set()
        phase_counts = {}
        for phase in ('e1', 'e2', 'e3'):
            paths = sorted((source / stage / phase).glob('*.json'))
            phase_counts[phase] = len(paths)
            for p in paths:
                record = json.loads(p.read_text(encoding='utf-8'))
                q = qs[p.stem]
                assert record['case'] == q and record['protocol_signature'] == sig
                for row in record['rows']:
                    key = row['case_id'], row['method'], row['prompt']
                    assert key not in seen
                    seen.add(key)
                    good, _ = validate(q, [tuple(x) for x in row['path']])
                    assert good == row['valid']
                    if row['path']:
                        assert math.isclose(path_length(row['path']), row['length'], rel_tol=1e-12, abs_tol=1e-9)
                    if not row['method'].startswith('GPT'):
                        assert row['operation'] >= 1 and row['storage'] >= 1
                    rows.append(row)
        requests = 0
        prompt_tokens = 0
        completion_tokens = 0
        for p in (source / stage / 'raw').glob('*.json'):
            r = json.loads(p.read_text(encoding='utf-8'))
            q = qs[r['case_id']]
            if r['variant'] == 'first':
                body = payload(q, r['prompt_kind'])
            else:
                assert r['variant'] in ('E2C', 'E2F')
                first = json.loads((p.parent / f"{q['id']}-{r['prompt_kind']}-first.json").read_text(encoding='utf-8'))
                body = correction_payload(q, r['prompt_kind'], text_of(first), r['variant'] == 'E2F')
            assert digest(body) == r['payload_sha256'] == digest(r['payload'])
            assert digest(r['response']) == r['response_sha256']
            assert r['response']['model'] == 'gpt-3.5-turbo-0125'
            requests += 1
            prompt_tokens += r['response']['usage']['prompt_tokens']
            completion_tokens += r['response']['usage']['completion_tokens']
        allrows[stage] = rows
        checks[stage] = {'expected_cases_per_phase': len(qs), 'saved_cases_by_phase': phase_counts,
                         'verified_rows': len(rows), 'verified_api_responses': requests,
                         'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}
        print(json.dumps({'audited': stage, **checks[stage]}), flush=True)
    assert checks['main']['saved_cases_by_phase']['e1'] == 1000
    assert checks['main']['saved_cases_by_phase']['e2'] == 1000
    reference = json.loads((root / 'improvement_reference.json').read_text(encoding='utf-8'))
    parity = 0
    for r in allrows['main']:
        if r['method'] in ('A', 'D', 'N'):
            for metric in ('operation', 'storage', 'length', 'valid'):
                assert math.isclose(r[metric], reference[r['case_id']][r['method']][metric], rel_tol=1e-12, abs_tol=1e-9)
            parity += 1
    assert parity == 3000
    summaries = summary(allrows['main'])
    write_csv(out / 'main_completed_conditions.csv', summaries)
    lookup = {(r['case_id'], r['method'], r['prompt']): r for r in allrows['main']}
    from improve_report import gm
    comparisons = []
    for kind in ('standard', 'cot', 'repe'):
        for treatment, control in [('E1', 'B'), ('E1', 'N'), ('E2F', 'B'), ('E2F', 'E2C'), ('E3', 'B'), ('E3', 'N')]:
            pairs = [(r, lookup[r['case_id'], control, None if control == 'N' else kind])
                     for r in allrows['main'] if r['method'] == treatment and r['prompt'] == kind]
            result = {'prompt': kind, 'treatment': treatment, 'control': control, 'n': len(pairs)}
            for metric in ('operation', 'storage', 'length'):
                eligible = [(a, b) for a, b in pairs if a['valid'] and b['valid']]
                result[metric + '_paired_ratio'] = gm([a[metric] / b[metric] for a, b in eligible])
                result[metric + '_paired_n'] = len(eligible)
            comparisons.append(result)
    write_csv(out / 'paired_completed_conditions.csv', comparisons)
    audit = {'experiment_complete': False, 'verified_saved_artifacts': True,
             'created_at': datetime.now(timezone.utc).isoformat(), 'failure_status': status,
             'protocol_signature': sig, 'checks': checks, 'legacy_classical_parity_cases': parity,
             'total_verified_rows': sum(c['verified_rows'] for c in checks.values()),
             'total_saved_api_responses': sum(c['verified_api_responses'] for c in checks.values()),
             'main_tables': summaries, 'main_paired': comparisons,
             'remaining': 'Finish E2 scaling, then full verification/report. Resume frozen runner; no new settings.',
             'e3_replay': json.loads((out / 'replay_audit.json').read_text(encoding='utf-8'))}
    assert audit['e3_replay']['complete'] and audit['e3_replay']['additional_api_requests'] == 0
    assert checks['main']['saved_cases_by_phase']['e3'] == 1000
    assert checks['scaling']['saved_cases_by_phase']['e3'] == 100
    save(out / 'audit.json', audit)
    display = [dict(r, method=LABEL[r['method']], prompt=PROMPT[r['prompt']]) for r in summaries]
    columns = [('method', '방법'), ('prompt', '프롬프트'), ('operation_ratio', '연산량 / A* %'),
               ('storage_ratio', '저장량 / A* %'), ('length_ratio', '길이 / A* %'),
               ('valid_ratio', '유효율 %'), ('n', '문제 수')]
    pc = [('prompt', '프롬프트'), ('treatment', '실험군'), ('control', '대조군'),
          ('operation_paired_ratio', '연산량 비율 %'), ('storage_paired_ratio', '저장량 비율 %'),
          ('length_paired_ratio', '길이 비율 %')]
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT 개선 실험 부분 결과</title><style>body{font:16px/1.65 "Malgun Gothic",sans-serif;background:#f4f6f8;color:#172033;margin:0}main{max-width:1180px;margin:30px auto;background:white;padding:36px}h1{font-size:30px}h2{margin-top:32px;color:#164e63}.notice{background:#fff3db;border-left:5px solid #b45309;padding:16px}table{border-collapse:collapse;width:100%;font-size:12px;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3f6}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px;font-size:12px}</style><main><h1>GPT 개선 실험 — 부분 결과 보고서</h1>'''
    document += '<div class="notice"><strong>확장성 평가 일부가 미완료인 부분 보고서입니다.</strong> OpenAI API가 credit_balance_exhausted / insufficient_quota를 반환해 실험 2 확장성 평가 중 실행이 중단됐습니다. 실험 3은 이미 저장된 응답·실행 기록으로 추가 API 호출 없이 재생 평가했습니다. 아래 본 평가 표는 세 실험의 완료된 1,000문제 전체 결과이며, 실험 2의 미완료 확장성 사례를 성공한 것으로 처리하지 않았습니다.</div>'
    document += '<h2>완료 범위와 평가 규모</h2><p>기존과 동일한 지도 100개 × 출발·도착점 쌍 10개 = 1,000문제에 Few-shot·CoT·RePE를 각각 적용했습니다. 따라서 개선안 하나의 본 평가는 3,000개입니다. 이는 학습 step 수가 아닙니다.</p>'
    document += '<pre>' + html.escape(json.dumps(checks, ensure_ascii=False, indent=2)) + '</pre>'
    document += '<h2>완료된 본 평가에서의 해석</h2><p><strong>실험 1:</strong> 경유점 정제는 기존 LLM-A*보다 연산량을 줄였지만, GPT 경유점 없는 g+2h 대조군보다 연산량이 낮지는 않았습니다. 개선의 일부를 잘못된 GPT 안내를 배제하는 효과로 해석할 수 있으며, GPT 안내의 추가 기여가 입증됐다고 주장하지 않습니다.</p>'
    document += '<p><strong>실험 2:</strong> 오류 피드백으로 GPT 자체 출력의 유효율은 상승했지만, 같은 호출 횟수의 비구체적 재검토 대조군 대비 최종 탐색 성능은 일관되게 좋아지지 않았습니다. 출력 품질 개선과 최종 탐색 성능 개선을 구분해야 합니다.</p>'
    e3 = [r for r in summaries if r['method'] == 'E3']
    document += '<p><strong>실험 3:</strong> 최초 응답 수집 때 저장한 조건부 호출 실행 기록과, 동일 응답을 사용하는 재생 결과가 연산량·저장량·길이·유효성·전체 경로·경유점·호출 시점·호출 횟수에서 일치함을 확인했습니다. 호출 수는 프롬프트별 ' + ', '.join(PROMPT[r['prompt']] + ' ' + str(r['api_calls']) + '/1000' for r in e3) + '입니다. API 사용량과 탐색 성능을 별도로 비교해야 합니다.</p>'
    document += '<p>실행 순서의 예외: 크레딧 부족으로 실험 2 확장성이 중단된 후, 추가 요청이 필요 없는 실험 3 재생을 먼저 마쳤습니다. 고정된 조작변인·입력·평가 규칙은 변경하지 않았습니다. 원래 계획의 모든 단계가 순서대로 완료됐다고 주장하지 않습니다.</p>'
    document += '<h2>본 평가 결과</h2><p>연산량·저장량·길이는 동일 문제 A* 대비 비율의 기하평균입니다. 유효한 양수 측정치를 집계하며 유효율의 분모는 전체 문제입니다. 세 개선안을 결합하지 않았습니다.</p>' + table(display, columns)
    document += '<h2>짝지은 직접 비교</h2><p>100% 미만은 해당 대조군보다 감소했다는 뜻입니다. E2F는 구체적 오류 피드백, E2C는 같은 횟수의 추가 호출 대조군입니다. N은 GPT 경유점 없는 g+2h입니다.</p>' + table(comparisons, pc)
    document += '<h2>통제 조건과 검증</h2><p>모델 gpt-3.5-turbo-0125, temperature=0, max_tokens=1000, 최초 프롬프트·지도·경계·충돌 판정·탐색 규칙을 유지했습니다. 새 최초 응답을 실험군과 대조군에서 공유했습니다. 사용자 요청에 따라 실행 호스트는 Windows에서 Linux 서버로 이동했고, 기존 재현의 A*·Dynamic Weighted A*·g+2h 지표 3,000건과 일치함을 확인했습니다. GPT 추론은 API에서, 탐색은 서버 CPU에서 실행했습니다.</p>'
    document += '<p>저장된 응답의 요청·응답 해시, 반환 모델명, 경로 유효성 및 길이, 코드·프로토콜 해시를 검증했습니다. 평가 중간 결과를 보고 설정을 조정하지 않았습니다. 지연시간이나 독립된 새 데이터에 대한 일반화 우위는 주장하지 않습니다.</p>'
    document += '<h2>중단 원인과 재개</h2><pre>' + html.escape(json.dumps(status, ensure_ascii=False, indent=2)) + '</pre>'
    document += '<p>API 크레딧 문제가 해결되면 동일한 실행기를 다시 시작합니다. 저장된 응답과 평가를 재사용하고, 누락된 실험 2 확장성 평가를 이어서 실행한 다음 완전한 결과를 검증합니다.</p><p><a href="main_completed_conditions.csv">완료 조건 CSV</a> · <a href="paired_completed_conditions.csv">직접 비교 CSV</a> · <a href="audit.json">검증 기록</a></p></main></html>'
    (out / 'report.html').write_text(document, encoding='utf-8')
    entries = {str(p.relative_to(root)).replace('\\', '/'): p for p in out.iterdir()
               if p.is_file() and p.name != 'export.json'}
    for p in source.rglob('*'):
        if p.is_file() and p.name != 'runner.pid':
            entries['results/improvements_partial/evidence/' + p.relative_to(source).as_posix()] = p
    manifest = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in entries.items()}
    archive = root / 'improvement-results-partial.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, p in entries.items():
            z.write(p, name)
        z.writestr('archive_manifest.json', json.dumps(manifest, indent=2))
    export = {'archive': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'bytes': archive.stat().st_size, 'files': len(entries), 'experiment_complete': False}
    save(out / 'export.json', export)
    print(json.dumps(export), flush=True)


if __name__ == '__main__':
    main()
