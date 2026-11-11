"""Build the requested combined Markdown report from frozen experiment results."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/improvements_final/report.md'
OUT = OLD.with_name('final_report.md')
RESULT = ROOT / 'results/candidates_v1'
SUMMARY = RESULT / 'report/summary.json'
CASES = RESULT / 'report/per_case.csv'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def link(label, path):
    return f'[{label}]({path.as_posix()})'


def build():
    old_hash = sha(OLD)
    original = OLD.read_text(encoding='utf-8-sig').strip()
    data = read_json(SUMMARY)
    verification = read_json(RESULT / 'local_verification.json')
    protocol = read_json(RESULT / 'protocol.json')
    calibration = read_json(RESULT / 'calibration.json')
    assert verification['verified'] and verification['total_rows'] == 4100
    assert verification['all_paths_revalidated'] and not verification['bound_violations']
    summary = {(r['stage'], r['method']): r for r in data['summary']}
    with CASES.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4100
    primary = [r for r in rows if r['stage'] == 'main']
    baseline = {r['case_id']: r for r in primary if r['method'] == 'FOCAL_EUCLIDEAN'}
    names = {
        'FOCAL_EUCLIDEAN': 'FOCAL 기준', 'CONFORMAL': 'Conformal',
        'CHOMP_STANDARD': 'CHOMP 계열 + LLM', 'FMM_LLM': 'Fast Marching + LLM',
        'FMM_UNIFORM': 'Fast Marching (LLM 없음)',
        'TRANSPATH_11': 'TransPath 계열 · seed 11',
        'TRANSPATH_22': 'TransPath 계열 · seed 22',
        'TRANSPATH_33': 'TransPath 계열 · seed 33',
        'ASTAR': 'A*', 'WEIGHTED_1_1': 'Weighted A* (1.1)',
    }
    methods = ['CONFORMAL', 'CHOMP_STANDARD', 'FMM_LLM', 'FMM_UNIFORM',
               'TRANSPATH_11', 'TRANSPATH_22', 'TRANSPATH_33']

    def ci(row, key, digits=3):
        return (f"{row[key + '_ratio']:.{digits}f} "
                f"[{row[key + '_ci_low']:.{digits}f}, {row[key + '_ci_high']:.{digits}f}]")

    def effect(ratio):
        return f'{abs(1-ratio)*100:.2f}% ' + ('감소' if ratio < 1 else '증가' if ratio > 1 else '동일')

    main_table = table(
        ['방법', '확장 수 비율 [95% CI]', '탐색량 변화', '로컬 시간 비율 [95% CI]', '길이 / 최적'],
        [[names[m], ci(summary['main', m], 'expansions', 5 if m == 'CONFORMAL' else 3),
          effect(summary['main', m]['expansions_ratio']), ci(summary['main', m], 'local_cold'),
          f"{summary['main', m]['length_ratio']:.4f}"] for m in ['FOCAL_EUCLIDEAN'] + methods])

    diagnostics = {}
    distribution_rows = []
    for m in methods:
        rr = [r for r in primary if r['method'] == m]
        x = [int(r['expansions']) for r in rr]
        b = [int(baseline[r['case_id']]['expansions']) for r in rr]
        ratio = sum(x) / sum(b)
        d = dict(wins=sum(a < c for a, c in zip(x, b)),
                 ties=sum(a == c for a, c in zip(x, b)),
                 losses=sum(a > c for a, c in zip(x, b)),
                 zero_expansions=x.count(0), total_expansions_ratio=ratio)
        diagnostics[m] = d
        # Cross-check the frozen summary against the problem-level records.
        geometric = math.exp(sum(math.log(max(a, 1)/max(c, 1)) for a, c in zip(x, b))/len(x))
        assert math.isclose(geometric, summary['main', m]['expansions_ratio'], rel_tol=1e-12)
        assert math.isclose(sum(x)/len(x), summary['main', m]['expansions_mean'], rel_tol=1e-12)
        distribution_rows.append([names[m], f"{d['wins']} / {d['ties']} / {d['losses']}",
                                  f"{sum(x)/len(x):.2f}", effect(ratio), d['zero_expansions']])
    distribution_table = table(
        ['방법', '감소 / 동일 / 증가 문제 수', '평균 확장 수', '확장 수 합계 변화', '0회 확장 문제'],
        distribution_rows)

    paired_specs = [
        ('CONFORMAL', 'FIXED_TUBE_2', 'Conformal / 고정 폭 2', '보정 없는 좁은 영역보다 낫지만 FOCAL 대비 개선은 없음'),
        ('CHOMP_STANDARD', 'VGATE_STANDARD', 'CHOMP / V-Gate', '같은 LLM 입력의 거부 방식 대비 제한적 개선'),
        ('CHOMP_STANDARD', 'CHOMP_STRAIGHT', 'CHOMP LLM 초기화 / 직선 초기화', '본실험에서 LLM 초기화의 추가 기여 신호'),
        ('FMM_LLM', 'FMM_UNIFORM', 'FMM+LLM / FMM 균일 속도', 'LLM의 추가 개선은 확인되지 않음'),
    ]
    paired_table = table(['본실험 비교', '확장 수 비율 [95% CI]', '해석'], [
        [label, f"{data['paired']['main_'+m+'_vs_'+b]['expansion_ratio']:.3f} "
                f"[{data['paired']['main_'+m+'_vs_'+b]['ci'][0]:.3f}, "
                f"{data['paired']['main_'+m+'_vs_'+b]['ci'][1]:.3f}]", interpretation]
        for m, b, label, interpretation in paired_specs])

    def secondary_table(stage, chosen):
        return table(['방법', '확장 수 비율 [95% CI]', '탐색량 변화', '로컬 시간 비율 [95% CI]'], [
            [names[m], ci(summary[stage, m], 'expansions'), effect(summary[stage, m]['expansions_ratio']),
             ci(summary[stage, m], 'local_cold')] for m in chosen])

    times_table = table(['방법', '방법별 준비 ms', '탐색 ms', '그래프 포함 총 ms', '필요한 과거 LLM 응답 수'], [
        [names[m], f"{summary['main', m]['preparation_ms_mean']:.2f}",
         f"{summary['main', m]['search_ms_mean']:.2f}",
         f"{summary['main', m]['local_cold_ms_mean']:.2f}",
         summary['main', m]['llm_responses_required']]
        for m in ['FOCAL_EUCLIDEAN'] + methods + ['ASTAR', 'WEIGHTED_1_1']])
    training_table = table(['시드', '학습 epoch', '선택 epoch', 'GPU 학습 시간 초', '파라미터 수'], [
        [seed, r['epochs'], r['best_epoch'], f"{r['seconds']:.2f}", f"{r['parameters']:,}"]
        for seed, r in data['training'].items()])

    intro = r'''# LLM-A\* 개선 실험 통합 최종 보고서 — 기존 3개 개선안과 추가 4개 후보

작성일: 2026-10-14 · 기존 28,270행 + 추가 4,100행의 검증 결과를 통합

**추가 네 후보 중 탐색량 감소 근거가 가장 뚜렷한 것은 Fast Marching입니다. CHOMP 계열은 일부 문제에서 제한적인 개선을 보였습니다. 그러나 준비 비용을 포함한 전체 실행시간의 일관된 단축은 네 후보 모두 확인되지 않았습니다.**

| 추가 후보 | 본실험의 핵심 결과 | 최종 판단 |
| --- | --- | --- |
| Conformal Prediction | 확장 수 비율 약 1.000, 로컬 시간 1.26배; 후보 영역이 자유 칸의 93.6% | 포함률은 높지만 탐색 효율 개선 없음 |
| CHOMP 계열 수리 | 문제별 확장 수 비율 기하평균 10.8% 감소, 확장 수 합계 1.0% 감소, 로컬 시간 5.26배 | 소수 문제의 수리·조기 반환에 따른 제한적 개선 |
| Fast Marching + LLM | 확장 수 비율 기하평균 25.6% 감소, 확장 수 합계 25.7% 감소, 로컬 시간 1.79배 | 탐색량 개선의 가장 뚜렷한 근거; 준비 비용 상쇄 실패 |
| TransPath 계열 학습 | 본실험의 확장 수 비율 기하평균 86.9~102.8% 증가; 새 미로에서는 8.5~17.0% 감소 | 본실험 개선 실패, 지도 유형에 따른 효과 차이 |

위 배율·감소율은 추가 실험의 같은 문제에 대한 FOCAL 기준 대비 값입니다. 시간 배율도 문제별 비율의 기하평균이며, 평균 밀리초를 나눈 값과 다릅니다. Fast Marching과 CHOMP의 개별 95% 지도 bootstrap 구간은 본실험 확장 수 비율 1 미만이지만, 다중 비교 보정과 독립 재현을 거치지 않은 탐색적 근거입니다. “통계적으로 확증된 일반적 우월성”으로 해석하지 않습니다.

기존 3개 개선안의 결과·표·한계는 1부에 보존하고, 추가 4개 후보의 설계·결과·판단을 2부에 이어 적었습니다. 1부의 “신뢰구간을 산출하지 않았다”는 설명은 기존 세 개선안에만 해당합니다. 두 실험은 대조군과 탐색량 정의가 달라 감소율을 한 순위로 합치지 않습니다.

## 1부. 기존 세 개선안 보고서

다음은 첨부 보고서의 본문입니다. 통합 문서에 맞춰 제목 수준만 조정했으며 수치와 해석은 유지했습니다.

'''
    old_embedded = re.sub(r'^(#+) ', r'\1# ', original, flags=re.MULTILINE)
    # The original title becomes a subsection under Part 1.
    old_embedded = old_embedded.replace('## LLM-A\\* 개선 실험 최종 보고서',
                                        '### LLM-A\\* 개선 실험 최종 보고서', 1)
    # Original numbered sections are also level 3; their nested method headings become level 4.

    cp = data['conformal']['main']
    new = f'''

## 2부. 추가 네 후보의 GPU 서버 실험

### 7. 실험 범위와 비교 기준

기존에 사용한 Sirius 서버의 NVIDIA RTX A5000에서 학습·신경망 추론을 수행했습니다. 격자 그래프 생성, CHOMP 수리, Fast Marching과 탐색은 CPU에서 수행했습니다. 실행 경로는 `/home/work/fanimal_hidream/candidates-20261014`입니다. Python 3.12.3, NumPy 1.26.4, PyTorch 2.7.0a0+7c8ec84dab.nv25.03, CUDA 12.8 환경이며 세부 버전은 환경 기록에 보존했습니다.

| 구분 | 기존 세 개선안 | 이번 네 후보 |
| --- | --- | --- |
| 주 대조군 | 기본 LLM-A* B; 별도 GPT 없는 g+2h N | 동일 구현의 FOCAL_EUCLIDEAN, w=1.1 |
| 탐색량 | 중복 포함 heap-pop 수 | 실제 노드 확장 수; heap-pop과 구분 |
| 본실험 | 1,000문제 × 3프롬프트 | 평가용 20지도 × 10문제 = 200문제 |
| LLM 응답 | 새 수집 후 조건 간 공유·캐시 | 기존 검증 응답 재사용, 신규 API 호출 0회 |
| 통계 | 문제별 대응 비율의 기하평균, 신뢰구간 없음 | 대응 비율 기하평균 + 지도 단위 bootstrap 95% 구간 |
| 시간 | 실제 지연시간 우위를 주장하지 않음 | 그래프 생성·준비·추론·탐색 측정; 과거 API 지연 별도 |

추가 후보의 기본 탐색기는 허용적 octile 하한을 사용하는 FOCAL이며 `w=1.1`입니다. 후보 안내는 FOCAL 내부의 우선순위에 사용하고, Conformal 영역 밖 노드를 제거하지 않습니다. CHOMP의 검증된 수리 경로는 실행 가능한 상한으로도 사용합니다. A*와 Weighted A*(1.1), 고정 폭 영역, 수리 없는 V-Gate, 직선 초기화 수리, LLM 없는 FMM, 학습 전 신경망도 대조군에 포함했습니다.

#### 7.1 지도 단위 분리와 평가 규모

| 역할 | 지도 수 | 문제 수 | 용도 |
| --- | --- | --- | --- |
| 학습 | 50 | 500 | TransPath 계열 모델 학습 |
| 검증 | 10 | 100 | 검증 BCE가 최소인 체크포인트 선택 |
| 보정 | 20 | 200 | Conformal 보정값 계산 |
| 본실험 | 20 | 200 | 15개 조건, 3,000행 |
| 새 미로 OOD | 20 | 100 | 지도당 5문제, LLM 없는 8개 조건, 800행 |
| 확대 평가 | 본실험 지도 중 10 | 20 | 지도별 첫 문제를 2배·4배 확대, 15개 조건, 300행 |

분할 시드 `{protocol['protocol']['split_seed']}`로 같은 지도의 10문제를 함께 배치했습니다. 프롬프트 예시 지도 5개는 학습 집합에만 넣었습니다. 학습·검증·보정·본실험 지도는 서로 겹치지 않습니다. 다만 본실험 원본 지도와 GPT 응답은 앞선 연구에서 이미 분석한 자료이므로, **이번 학습·보정에서 제외한 평가 집합이지 연구 전체에서 한 번도 보지 않은 완전한 독립 벤치마크는 아닙니다.**

총 4,100행 모두 경로 유효성과 최적 길이의 1.1배 이내 조건을 통과했습니다. 이는 실패 시 기본 탐색을 사용하는 전체 알고리즘의 결과이며, 원래 GPT 안내나 수리·학습 모델이 모든 문제에서 성공했다는 뜻은 아닙니다.

#### 7.2 실제 구현한 범위

| 후보 | 이번 구현 | 해석 범위 |
| --- | --- | --- |
| Conformal | 세 LLM 제안 경로 주변의 격자 영역을 지도 단위 점수로 보정; 영역 밖 거리로 FOCAL 우선순위 부여 | 세 경로 중 정답 경로를 분류하는 방식이 아닌 공간 후보 집합 변형 |
| CHOMP 계열 | 48개 경로 표본, 최대 160회 사전조건화 경사 갱신; 평활도·장애물 비용 최적화 후 8방향 격자 경로와 모든 이동 재검증 | CHOMP 아이디어의 2D 변형이며 원 논문 구현의 정확 재현 아님 |
| Fast Marching | 전 격자의 1차 upwind Eikonal 도착 시간; 균일 속도 1과 LLM 경로 반경 2 안의 속도 2 조건 비교 | 전체 안내 지도 생성 비용을 포함한 FOCAL 보조 안내 |
| TransPath 계열 | 7채널 입력, 24/48/64채널 합성곱 인코더, 2개 Transformer 층·4 heads·FFN 128, 합성곱 디코더 | 자체 소형 모델이며 공식 TransPath 코드·사전학습 모델의 성능 검증 아님 |

CHOMP는 Few-shot(`standard`) 한 응답으로 초기화했습니다. Conformal과 FMM+LLM은 Few-shot·CoT·RePE의 세 응답을 함께 사용했습니다. 따라서 모든 후보가 같은 수의 LLM 응답을 필요로 하는 것은 아닙니다.

### 8. 지표와 통계 해석

문제별 확장 수 비율의 기하평균은 `exp(mean(log(max(E방법,1)/max(E기준,1))))`입니다. 1보다 작으면 감소이며 감소율은 `100 × (1 − 비율)`입니다. 검증된 상한 경로를 즉시 반환해 확장이 0인 경우 로그 비율 계산에만 1을 사용했습니다. 산술평균·합계·문제 수 집계에는 실제 0을 유지했습니다.

지도 안의 문제별 로그 비율을 먼저 평균하고, 지도 평균들을 복원추출한 2,000회 bootstrap의 2.5·97.5백분위수를 95% 구간으로 사용했습니다. bootstrap 시드는 2026이며 본실험·OOD는 각각 20지도, 확대 평가는 10지도를 표본 단위로 삼았습니다. 같은 지도 안의 문제들을 독립 표본으로 취급하지 않았습니다.

표의 신뢰구간은 **개별 비교에 대한 구간**입니다. 다중 비교 보정이나 별도 p값 검정은 수행하지 않았습니다. 구간 전체가 1 미만인 결과는 해당 지표의 감소를 지지하는 탐색적 근거이며, 방법 전체의 일반적 우월성을 확증하지 않습니다. 구간에 1이 들어가면 개선을 분명하게 확인하지 못한 것입니다. 이것이 두 방법의 동등함을 증명하는 것은 아닙니다.

로컬 시간은 매 문제의 그래프 생성 + 방법별 준비·GPU 추론 + 실제 탐색입니다. 모델은 메모리에 상주하고 warm-up을 마친 상태이며 입출력 전송과 CUDA 동기화는 포함했습니다. 파일의 `local_cold`라는 이름은 그래프를 새로 만드는 조건을 뜻하며, 모델 로딩을 포함한 완전한 서비스 최초 기동 시간은 아닙니다. API 지연·학습·정답 생성·보정·warm-up은 별도입니다.

### 9. 본실험 결과 — 같은 200문제의 FOCAL 대비

{main_table}

모든 행의 경로 유효율은 100%입니다. 길이 비율은 최적 경로 대비 기하평균이며 1.1 상한은 개별 경로에서 검사했습니다. 주 대조군의 평균 확장 수는 {summary['main', 'FOCAL_EUCLIDEAN']['expansions_mean']:.3f}회입니다. Conformal의 아주 작은 차이가 반올림으로 사라지지 않도록 해당 행만 소수점 다섯 자리로 표시했습니다.

![추가 네 후보의 본실험 비교]({(RESULT / 'report/comparison.png').as_posix()})

#### 9.1 평균 효과와 실제 개선 문제 수

다음은 원자료를 추가 집계한 기술 통계입니다. 확장 수 합계 변화는 같은 200문제에서 실제 확장 횟수를 합한 값의 비율이며, 문제별 기하평균과는 다른 질문에 답합니다. 이 표의 합계 변화에는 별도 신뢰구간을 산출하지 않았습니다.

{distribution_table}

CHOMP의 10.8%는 200문제 전체 확장 수의 10.8% 절감을 뜻하지 않습니다. 182문제에서 기준과 같고 13문제에서만 감소했으며, 7문제는 준비 단계에서 얻은 경로로 확장 없이 종료했습니다. 합계 절감은 약 1.0%이므로 소수 문제의 큰 상대적 감소와 0회 처리 규칙의 영향을 함께 보아야 합니다. TransPath 계열 역시 기하평균 비율은 1.87~2.03배이지만 실제 확장 수 합계 증가는 약 6.7~8.8%입니다.

#### 9.2 후보별 해석

**① Conformal Prediction: 포함률을 확보했으나 효율 개선은 없었습니다.**

보정은 α=0.1, 20지도 점수의 19번째 순서통계량 {calibration['quantile']:.4f}를 사용했습니다. 점수는 지도 안 10문제의 고정된 기준 최적 경로 전체에서 세 제안 경로의 합집합까지의 최대 거리를 제안 간 분산 척도로 나눈 값입니다. 평가 때는 정답 없이 제안만으로 분산 척도와 후보 영역을 만듭니다.

본실험에서는 200/200문제의 기준 최적 경로가 전부 포함됐고, 지도 내 10경로 동시 포함도 20/20지도였습니다. 그러나 평균 후보 영역이 자유 칸의 {100*cp['average_set_fraction']:.2f}%여서 우선순위를 구별하는 힘이 거의 없었습니다. 확장 수는 약 0.01% 감소에 그쳤고 시간 비율은 1.26배였습니다. 20지도에서 관측한 100%를 모집단 100% 보장으로 해석할 수 없습니다. 이론적 90% 포함 목표는 적격 지도들의 교환가능성 가정 아래이며, 모든 최적 경로가 아닌 문제별로 고정한 하나의 최적 경로에 대한 것입니다. OOD·확대 분포에는 같은 보장을 주장하지 않습니다.

**② CHOMP 계열 수리: 복구 가능성은 보였으나 실용적 이득은 작았습니다.**

처음 무효였던 188개 LLM 안내 중 6개를 복구했습니다(3.19%). 원래 유효했던 12개를 포함해 수리 후 경로가 승인된 경우는 18/200개입니다. 격자 변환 후 충돌·경계를 다시 검사한 경로만 사용했습니다. FOCAL 대비 확장 수 비율의 개별 95% 구간은 0.810~0.970으로 감소를 지지하지만, 실제 확장 수 합계 절감은 1.03%이고 수리 계산 때문에 로컬 시간은 5.26배였습니다. 경로 길이 비율도 기준 1.0101에서 1.0131로 소폭 증가했습니다. 수리가 큰 방향 오류를 넓게 해결했다고 볼 수는 없습니다.

**③ Fast Marching: 탐색량 감소는 뚜렷하지만 준비 비용과 LLM 추가 기여가 남았습니다.**

FMM+LLM의 확장 수는 기하평균 기준 25.58% 감소했습니다. 비율 구간 0.661~0.823은 약 17.7~33.9% 감소에 해당하며, 확장 수 합계도 25.72% 감소했습니다. 경로 길이 비율은 1.0059였습니다. 이번 본실험에서 가장 설득력 있는 탐색량 개선입니다.

그러나 LLM 없는 균일 속도 FMM도 24.95%의 기하평균 감소를 보였습니다. FMM+LLM / FMM 균일 속도 비율은 0.992 [0.962, 1.026]으로 LLM의 추가 이득을 확인하지 못했습니다. 안내 지도 구성 비용을 포함하면 각각 로컬 시간은 기준의 1.79배와 1.58배입니다. **현재 증거는 FMM 안내가 뒤따르는 탐색의 확장 수를 줄였다는 것이며, 전체 계산이나 온라인 시간을 줄였다는 것은 아닙니다.**

**④ TransPath 계열 학습: 본실험에서는 악화됐고 새 미로에서만 탐색량이 줄었습니다.**

세 학습 시드 모두 본실험 확장 수 비율이 1보다 컸고 개별 95% 구간도 1을 넘었습니다. 시간은 약 1.25~1.26배였습니다. seed 22만 골라 성과로 제시하지 않고 세 시드를 모두 보고합니다. 새 미로에서는 세 시드 모두 확장 수 감소를 보였지만, 본실험 실패와 지도 유형에 따른 차이를 함께 보면 일반적 개선이라고 할 수 없습니다. 이는 이번 소형 모델·학습량·레이블·FOCAL 결합의 결과이며 공식 TransPath 방법 전체에 대한 부정적 결론은 아닙니다.

#### 9.3 같은 계열 안의 대조 실험

{paired_table}

고정 폭 영역은 FOCAL 대비 확장 수 비율이 약 1.197로 오히려 나빴습니다. 따라서 Conformal이 고정 폭보다 좋아진 것은 잘못된 좁은 안내의 손해를 줄였다는 근거이며, 기본 탐색보다 좋아졌다는 근거는 아닙니다. CHOMP의 직선 초기화는 처음 무효인 200개 중 6개를 복구했습니다. 본실험의 LLM 초기화 우위는 확대 평가에서는 유지되지 않았습니다.

### 10. 새 미로와 확대 지도 결과

#### 10.1 분포 밖 새 미로 — 20지도, 100문제

{secondary_table('ood', ['FMM_UNIFORM', 'TRANSPATH_11', 'TRANSPATH_22', 'TRANSPATH_33'])}

TransPath 계열의 확장 수 감소는 8.48~16.95%이고 각 개별 구간이 1 미만입니다. 학습 전 동일 구조의 seed 11 모델은 비율 약 1.297로 악화돼, 이 미로 조건에서 학습이 도움이 된다는 신호가 있습니다. 하지만 seed 22의 시간 비율 0.989는 구간에 1이 포함되고, 다른 두 시드의 시간은 증가했습니다. 따라서 새 미로에서도 확실한 시간 단축은 확인되지 않았습니다. LLM 응답을 새로 수집하지 않았으므로 Conformal·CHOMP+LLM·FMM+LLM은 이 평가에서 실행하지 않았습니다.

#### 10.2 2배·4배 확대 — 10지도, 20문제

{secondary_table('scaling', methods)}

FMM+LLM과 FMM 균일 속도는 각각 32.74%, 54.58%의 기하평균 확장 수 감소를 보였으나 시간은 1.57배·1.38배였습니다. 이 표본에서도 LLM을 더한 방식이 균일 속도보다 좋다는 근거는 없습니다. CHOMP의 확장 수 비율은 0.635지만 구간 상한이 1이며, 무효 입력 20개 중 복구는 LLM 초기화 2개·직선 초기화 4개였습니다. TransPath 세 시드의 시간 구간도 모두 1을 포함합니다.

확대 평가는 기존 LLM 응답의 좌표를 지도와 함께 확대했습니다. GPT가 큰 지도를 새로 해석한 결과가 아니며, 20문제는 10개 원본 지도에서 파생된 작은 표본입니다. Conformal 포함률은 관측상 100%였지만 확대 분포에 대한 보장은 아닙니다.

### 11. 준비·학습·실행 비용

#### 11.1 본실험의 산술평균 시간

{times_table}

공통 그래프 생성 비용은 문제당 평균 약 136.14ms이며 위 총 시간에 포함됩니다. 같은 문제에서는 같은 그래프 생성 측정값을 모든 조건에 적용했습니다. 일부 퇴화한 기하 조건에서의 대체 판정 처리 비용까지 포함한 값이어서 산술평균 총 시간에 큰 영향을 줍니다. 각 방법의 준비·탐색 ms는 별도 항목입니다.

예를 들어 CHOMP의 평균 총 시간은 255.07/141.48 ≈ 1.80배이지만, 문제별 시간 비율의 기하평균은 5.26배입니다. 두 수치는 집계 방식이 달라 모순이 아닙니다. 마찬가지로 FMM+LLM은 평균 총 시간의 비로는 약 1.11배, 문제별 기하평균 비율로는 1.79배입니다.

추가 실험에서 신규 API 호출은 없었습니다. CHOMP는 운영 시 문제당 과거 응답 1개에 해당하는 안내가 필요하며, 해당 자료의 기록된 평균 LLM 지연은 0.86초입니다. Conformal·FMM+LLM은 3개 응답이 필요하며 과거 지연 합의 평균은 5.69초입니다. 이 값은 이번 서버에서 새로 측정한 온라인 API 지연이 아니므로 위 시간에 섞지 않았습니다. LLM 없는 방법과 실제 서비스 비용을 비교할 때는 응답 생성 비용·지연도 추가해야 합니다.

A*와 Weighted A*(1.1)의 본실험 로컬 시간 비율은 각각 0.888, 0.939로 FOCAL 기준보다 작았습니다. 따라서 FOCAL 내부에서의 안내 효과만으로 기존 탐색기 전반보다 빠르다고 주장할 수 없습니다.

#### 11.2 학습과 보정의 별도 비용

{training_table}

학습은 AdamW, 학습률 0.001, weight decay 0.0001, batch 32로 수행했습니다. 정답은 정확한 격자 최단거리에서 `d(start,x)+d(x,goal) ≤ optimal+0.5`를 만족하는 칸입니다. 시험 결과로 체크포인트를 선택하지 않고 검증 BCE만 사용했습니다.

세 시드 GPU 학습 합계는 {data['training_seconds']:.2f}초(약 {data['training_seconds']/60:.2f}분), 학습·검증 정답 준비는 {data['dataset_preparation']['seconds']:.2f}초와 Dijkstra 확장 {data['dataset_preparation']['label_dijkstra_expansions']:,}회, Conformal 보정은 {calibration['seconds']:.2f}초였습니다. 이 오프라인 비용은 문제별 로컬 시간에서 제외하고 별도로 보고합니다. 모델 로딩·warm-up, 데이터 이동·환경 설치까지 포함한 전체 프로젝트 소요시간과도 구분됩니다.

![TransPath 계열 세 시드 학습 곡선]({(RESULT / 'report/learning_curves.png').as_posix()})

### 12. 검증과 해석의 한계

- 서버·로컬에서 4,100개 경로를 모두 재검증했고 길이 상한 위반은 0건이었습니다. 아카이브 344개 파일의 해시를 대조했으며, 기존 기하 판정 기반 Dijkstra로 본실험 20·미로 20·확대 20문제의 최적 길이도 별도로 대조했습니다.
- 처음에는 서버 시스템 환경의 GEOS 공유 라이브러리 문제로 기하 라이브러리 실행이 막혔습니다. 학습·평가 시작 전에 시스템 PyTorch를 사용하는 별도 환경에 Shapely 2.2.0을 설치해 해결했습니다. 환경 변경은 감사 기록에 남겼습니다.
- 시간은 공유 서버에서 고정된 방법 순서로 문제당 한 번 측정했습니다. 지도 bootstrap 구간은 지도 간 변동을 나타내며 하드웨어 반복 측정의 불확실성을 대신하지 않습니다. Python의 OPEN 스캔을 사용하는 FOCAL 구현이므로 최적화된 제품 수준 구현의 속도를 대표하지 않습니다.
- 본실험 지도 20개, 확대 지도 10개로 지도 표본 수가 작습니다. 세 학습 시드도 독립 데이터셋 세 개를 뜻하지 않습니다. 다수 후보·지표·시드 비교에 대한 다중 비교 보정은 하지 않았습니다.
- 기존 자료 재사용에 따른 선택 편향 가능성, Conformal 영역 변형과 CHOMP·TransPath의 자체 구현 범위를 감안해야 합니다. 방법 이름이 같다고 원 논문 성능을 재현했다고 볼 수 없습니다.
- FMM 전처리의 재사용, CHOMP를 유망한 일부 문제에만 적용, 더 큰 학습 데이터 등은 이번에 검증하지 않았습니다. 따라서 가능한 후속 개선 방향과 이미 확인한 성과를 구분합니다.

### 13. 추가 실험 원자료와 재현 경로

- {link('고정 프로토콜과 지도 분할', RESULT / 'protocol.json')}
- {link('전체 통계 JSON', SUMMARY)} · {link('전체 조건 집계 CSV', RESULT / 'report/summary.csv')} · {link('문제별 결과 CSV', CASES)}
- {link('서버 자동 생성 상세 보고서', RESULT / 'report/report.html')} · {link('추가 후보 해석 HTML', ROOT / 'results/candidates_final/report.html')}
- {link('Conformal 보정 기록', RESULT / 'calibration.json')} · {link('학습·검증 정답 준비 기록', RESULT / 'data_manifest.json')}
- {link('서버 검증', RESULT / 'verification.json')} · {link('로컬 재검증', RESULT / 'local_verification.json')}
- {link('환경 정보', RESULT / 'environment.json')} · {link('패키지 버전', RESULT / 'environment_packages.json')} · {link('환경 변경 감사 기록', RESULT / 'setup_deviation.json')}
- {link('탐색·후보 구현', ROOT / 'reproduction/candidate_core.py')} · {link('실험 실행기', ROOT / 'reproduction/candidate_run.py')} · {link('고정 통계 집계 코드', ROOT / 'reproduction/candidate_report.py')}
- {link('서버 원자료 ZIP', ROOT / 'candidate-results-verified.zip')} · {link('서버 실행 완료 화면', ROOT / 'results/candidates_server_proof/complete.png')}

추가 실험 ZIP SHA-256: `{verification['archive_sha256']}`

추가 실험 고정 프로토콜 서명: `{protocol['signature']}`

## 3부. 일곱 개선안의 종합 판단

| 단계·후보 | 확인한 성과 | 현재 근거로 주장할 수 없는 내용 |
| --- | --- | --- |
| 기존 E1 경유점 정제 | 기본 LLM-A* 대비 heap-pop 감소 | GPT 없는 g+2h보다 우수함, 통계적 유의성 |
| 기존 E2 오류 피드백 | GPT 출력 유효율 소폭 증가 | 동일 호출 횟수 대조군 대비 일관된 탐색 개선 |
| 기존 E3 조건부 호출 | 기본 LLM-A* 대비 heap-pop·논리적 API 호출 감소 | GPT 없는 g+2h보다 적은 연산량, 실제 응답시간 우위 |
| 추가 Conformal | 기준 경로 포함 관측률 100%; 좁은 고정 영역의 손해 완화 | FOCAL 대비 효율 개선, OOD 포함률 보장 |
| 추가 CHOMP 계열 | 무효 안내 일부 복구; 본실험 확장 수 비율의 제한적 감소 | 광범위한 문제 개선, 전체 시간 단축 |
| 추가 Fast Marching | 본실험·확대에서 확장 수 감소의 가장 뚜렷한 근거 | 준비 포함 전체 속도 향상, LLM의 추가 우위 |
| 추가 TransPath 계열 | 새 미로에서 학습 후 확장 수 감소 | 본실험 또는 여러 지도 유형에 걸친 일관된 개선 |

이번 네 후보에서 **“유의미한 성능 향상이 있었는가”에 대한 답은 지표별로 다릅니다. 탐색 단계의 확장 수에는 Fast Marching의 뚜렷한 개선과 CHOMP의 제한적 개선이 있었지만, 준비 비용과 응답 생성까지 고려한 실제 속도 향상은 입증하지 못했습니다.** Conformal은 효율 개선을 보이지 않았고, TransPath 계열은 본실험에서 악화됐으나 새 미로의 탐색량에는 긍정적 신호가 있었습니다.

후속 검증의 우선순위는 LLM 없는 FMM을 포함한 전처리 재사용·비용 절감 비교, CHOMP의 적용 대상 선별, 학습 안내의 지도 유형별 실패 분석입니다. 이는 다음 실험의 제안이며 이번 성과로 계산하지 않습니다. 기존 세 개선안과 추가 네 후보를 같은 문제·탐색기·측정 규칙으로 다시 비교하기 전에는 일곱 방법의 통합 성능 순위를 제시하지 않습니다.

문서 작성 근거: {link('첨부된 기존 Markdown 원본', OLD)}와 위 검증 완료 자료. 원본 Markdown과 고정 실험 산출물은 변경하지 않았습니다. 이 통합본에서 추가한 합계·개선 문제 수는 문제별 CSV로부터 계산했습니다.
'''
    content = intro + old_embedded + new
    # Validate every local link/image referenced by the final document.
    targets = re.findall(r'\]\((C:/[^)]+)\)', content)
    missing = [p for p in targets if not Path(p).exists()]
    assert not missing, missing
    assert sha(OLD) == old_hash
    assert old_embedded in content
    OUT.write_text(content, encoding='utf-8')
    audit = dict(
        report=str(OUT), source_original_sha256=old_hash,
        summary_sha256=sha(SUMMARY), per_case_sha256=sha(CASES),
        final_report_sha256=sha(OUT), local_links_checked=len(targets),
        original_preserved=True, additional_rows=len(rows),
        main_distribution=diagnostics,
        note='Descriptive distribution checks only; no new experiment or multiplicity adjustment.',
    )
    OUT.with_suffix('.sources.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k != 'main_distribution'}, ensure_ascii=False))


if __name__ == '__main__':
    build()
