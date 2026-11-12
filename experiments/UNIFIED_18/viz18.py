"""Self-contained HTML visualisation of summary18.json (no normalisation, no composite score).

    python viz18.py   ->  results/run_20261112/tradeoff18.html
"""
import json
from u18 import read
from u18 import RUN18 as R

def pick(d):
    g = lambda k: (d.get(k) or {}) if isinstance(d.get(k), dict) else None
    r = lambda k: None if not d.get(k) else dict(v=d[k]['ratio'], lo=d[k]['ci95'][0], hi=d[k]['ci95'][1], gsd=d[k]['gsd'], dirs=d[k]['directions'])
    return dict(valid=d['valid'], cases=d['cases'], bound=d['bound'], viol=d['bound_violations'],
                time=r('planner_s_vs_llm'), e2e=r('e2e_s_vs_llm'), exp=r('expansions_vs_llm'), op=r('paper_operation_vs_llm'),
                pops=r('heap_pops_vs_llm'), stor=r('paper_storage_vs_llm'), open=r('peak_open_vs_llm'), len=r('length_vs_llm'),
                ms=d['planner_s_mean_sd'], e2e_s=d['e2e_s_mean_sd'], exp_n=d['expansions_mean_sd'], open_n=d['peak_open_mean_sd'],
                stor_n=d['paper_storage_mean_sd'], cv=d['planner_cv_median'],
                lstar=d['lstar_gm'], lsd=d['lstar_sd'], opt=d['optimal'], qn=d['quality_n'], p95=d['p95_excess'], mx=d['max_excess'],
                o5=d['over5'], o10=d['over10'], resp=d['responses'], tin=d['prompt_tokens'], tout=d['completion_tokens'], usd=d['usd_list'],
                api=d['api_s_mean_sd'])

def build():
    s = read(R/'summary18.json')
    main = []
    for k, d in s['main'].items():
        x = pick(d); c = d['vs_control']
        x.update(label=k, owner=d['owner'], cond=d['condition'], ctrl=d['control'], online=d['online_llm'],
                 ctl={m: (None if not c[m] else dict(v=c[m]['ratio'], lo=c[m]['ci95'][0], hi=c[m]['ci95'][1])) for m in c})
        main.append(x)
    ref = {m: pick(d) for m, d in s['reference'].items()}
    for m in ('A_GM', 'B_llm', 'B_llm_paper'):
        d = s['reference'][m]
        ref[m]['t1'] = dict(op=100*d['paper_operation_vs_astar']['ratio'], st=100*d['paper_storage_vs_astar']['ratio'],
                            ln=100*d['length_vs_astar']['ratio'], valid=100*d['valid']/d['cases'])
    data = dict(main=main, ref=ref, cases=s['all']['B_llm']['cases'])
    html = TEMPLATE.replace('/*DATA*/', json.dumps(data, ensure_ascii=False))
    (R/'tradeoff18.html').write_text(html, encoding='utf-8')
    print(R/'tradeoff18.html')

TEMPLATE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>18개 방법 트레이드오프</title>
<style>
/* Layout: one reading column; matrix + scatter grid + group tables. Ratios are vs original LLM-A* (=100%). */
:root{--bg:#f7f8fa;--panel:#ffffff;--fg:#1d2330;--muted:#5f6878;--line:#dde1e8;--accent:#2f5bd3;
 --good:#1f8a5b;--bad:#c2412d;--sm:#2f5bd3;--jy:#c07a12;--gm:#1f8a5b;--ref:#6b7280;
 --sans:"Inter","Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;--mono:"JetBrains Mono",Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11141a;--panel:#181c24;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2b313d;--accent:#7a9cf5;
 --good:#4cc38a;--bad:#f07a63;--sm:#7a9cf5;--jy:#e2a84a;--gm:#4cc38a;--ref:#9aa3b2;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11141a;--panel:#181c24;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2b313d;--accent:#7a9cf5;
 --good:#4cc38a;--bad:#f07a63;--sm:#7a9cf5;--jy:#e2a84a;--gm:#4cc38a;--ref:#9aa3b2;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.55 var(--sans);margin:0}
main{max-width:1240px;margin:0 auto;padding-block:28px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1{font-size:24px;margin:0;text-wrap:balance}h2{font-size:17px;margin:0 0 8px}
p.note{color:var(--muted);margin:4px 0 0;max-width:80ch}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;min-width:0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:12px;position:sticky;top:0;background:var(--panel)}
td:first-child,th:first-child,td.l,th.l{text-align:left}
.own{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.ci{color:var(--muted);font-size:11px;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:18px}
svg text{fill:var(--fg);font:11px var(--sans)}svg .ax{stroke:var(--line)}svg .tick{fill:var(--muted)}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:6px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tabs button{font:inherit;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style></head><body><main>
<header><h1>18개 방법 vs 원 LLM-A* 트레이드오프</h1>
<p class="note">공개 1,000문제 · 한 PC 한 측정 블록 · 51조건 × 5회 · SM 통합 규칙(지도 5-fold OOF). 모든 비율은 문제별 (방법 ÷ 원 LLM-A*, 통합 프롬프트)의 기하평균이며 100% 미만은 원 LLM-A*보다 작다는 뜻이다. 정규화·가중치·종합 점수는 적용하지 않았다. 색은 원 LLM-A*보다 나은지(초록)/나쁜지(빨강)만 보여 주며, 진하기는 배율(로그)을 모든 열에 같은 척도로 쓴 것이다.</p>
<div class="legend"><span><i class="own" style="background:var(--sm)"></i>SM (승민)</span><span><i class="own" style="background:var(--jy)"></i>JY (준연)</span><span><i class="own" style="background:var(--gm)"></i>GM (규민)</span><span><i class="own" style="background:var(--ref)"></i>원 LLM-A* (기준점)</span></div></header>

<section><h2>0. 원 LLM-A* 점수표 (논문 Table 1 형식, A* = 100)</h2><div class="scroll"><table id="t1"></table></div>
<p class="note">Operation = 주 탐색 OPEN pop 수, Storage = g가 부여된 상태 수, 경로 길이는 A* 대비, 문제별 기하평균. 논문(LLAMA3 few-shot) 44.59 / 64.02 / 102.47 / 100. 18개 방법의 비교 분모는 같은 응답을 공유하는 통합 프롬프트 LLM-A*다.</p></section>

<section><h2>한눈에 보기: 원 LLM-A* 대비 비율</h2><div class="scroll"><table id="matrix"></table></div>
<p class="note">셀 아래 작은 글씨는 95% 구간. "—"는 해당 엔진에서 같은 정의로 분리할 수 없는 값.</p></section>

<section><h2>트레이드오프 산점도</h2><div class="grid" id="scatters"></div></section>

<section><h2>세부 표 (7개 평가 항목)</h2><div class="tabs" id="tabs" role="toolbar"></div><div class="scroll"><table id="detail"></table></div><p class="note" id="dnote"></p></section>
</main>
<script>
const D=/*DATA*/;
const OWN={SM:'var(--sm)',JY:'var(--jy)',GM:'var(--gm)'};
const f=(x,d=1)=>x==null?'—':x.toFixed(d);
const P=r=>r?f(100*r.v)+'%':'—';
const PCI=r=>r?`${f(100*r.v)}%<span class="ci">[${f(100*r.lo)}, ${f(100*r.hi)}]</span>`:'—';
const MS=(t,k=1,d=2)=>t&&t[0]!=null?`${(k*t[0]).toFixed(d)} ± ${(k*t[1]).toFixed(d)}`:'—';
const BND=b=>!b?'없음':b==1?'최단':'≤'+b+'L*';
const B=D.ref.B_llm;
function shade(r,lowerBetter=true){ if(!r) return ''; const l=Math.log2(r.v); if(Math.abs(l)<0.01) return '';
  const good=lowerBetter? l<0 : l>0; const a=Math.min(Math.abs(l)/3,1)*0.55+0.08;
  return `background:color-mix(in srgb, var(${good?'--good':'--bad'}) ${Math.round(a*100)}%, transparent)`;}
function nameCell(m){return `<td class="l"><i class="own" style="background:${OWN[m.owner]}"></i><b>${m.label}</b> <span class="ci" style="display:inline">${m.cond}</span></td>`}
// table 1
(function(){const rows=[['A_GM','A*'],['B_llm','LLM-A* (통합 프롬프트)'],['B_llm_paper','LLM-A* (논문 원 프롬프트)']];
 let h='<tr><th>조건</th><th>Operation</th><th>Storage</th><th>Relative path length</th><th>Valid path</th><th>계획기 ms</th><th>총응답 s(추정)</th><th>최단 경로</th><th>최대 초과율</th></tr>';
 for(const [k,n] of rows){const r=D.ref[k];h+=`<tr><td>${n}</td><td>${f(r.t1.op,2)}</td><td>${f(r.t1.st,2)}</td><td>${f(r.t1.ln,2)}</td><td>${f(r.t1.valid,1)}</td><td>${MS(r.ms,1000)}</td><td>${MS(r.e2e_s)}</td><td>${r.opt}/${r.qn}</td><td>${f(r.mx,2)}%</td></tr>`}
 document.getElementById('t1').innerHTML=h;})();
// matrix
(function(){let h='<tr><th class="l">방법</th><th>계획기 시간</th><th>총응답시간</th><th>이웃 확장</th><th>Operation(논문)</th><th>Storage(논문)</th><th>OPEN 최대</th><th>경로 길이</th><th>보장</th><th>최단/1000</th><th>최대 초과</th><th>논리 응답</th><th>LLM 기여(시간, 대조군 대비)</th></tr>';
 for(const m of D.main){h+=`<tr>${nameCell(m)}<td style="${shade(m.time)}">${PCI(m.time)}</td><td style="${shade(m.e2e)}">${PCI(m.e2e)}</td><td style="${shade(m.exp)}">${PCI(m.exp)}</td>`+
  `<td style="${shade(m.op)}">${P(m.op)}</td><td style="${shade(m.stor)}">${P(m.stor)}</td><td style="${shade(m.open)}">${P(m.open)}</td><td style="${shade(m.len)}">${P(m.len)}</td>`+
  `<td>${BND(m.bound)}</td><td>${m.opt}</td><td>${f(m.mx,2)}%</td><td>${Math.round(m.resp).toLocaleString()}</td><td style="${shade(m.ctl.planner_s)}">${m.online?PCI(m.ctl.planner_s):'<span class="ci" style="display:inline">LLM 없음</span> '+P(m.ctl.planner_s)}</td></tr>`}
 h+=`<tr><td class="l"><i class="own" style="background:var(--ref)"></i><b>원 LLM-A*</b></td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>없음</td><td>${B.opt}</td><td>${f(B.mx,2)}%</td><td>${B.resp.toLocaleString()}</td><td>—</td></tr>`;
 document.getElementById('matrix').innerHTML=h;})();
// scatter
function scatter(title,xf,yf,xl,yl,{xlog=true,ylog=false,yfmt=v=>v.toFixed(3),xfmt=v=>(100*v).toFixed(0)+'%',ref=null}={}){
 const W=560,H=360,ml=62,mr=16,mt=26,mb=44;const pts=D.main.map(m=>({m,x:xf(m),y:yf(m)})).filter(p=>p.x!=null&&p.y!=null&&p.x>0);
 if(ref) pts.push(ref);
 const xs=pts.map(p=>xlog?Math.log10(p.x):p.x), ys=pts.map(p=>ylog?Math.log10(p.y):p.y);
 let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);const px=(x1-x0)*0.08||0.1,py=(y1-y0)*0.1||0.01;x0-=px;x1+=px;y0-=py;y1+=py;
 const X=v=>ml+((xlog?Math.log10(v):v)-x0)/(x1-x0)*(W-ml-mr), Y=v=>H-mb-((ylog?Math.log10(v):v)-y0)/(y1-y0)*(H-mt-mb);
 let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${title}"><text x="${ml}" y="16" style="font-weight:600;font-size:12px">${title}</text>`;
 const xt=[];if(xlog){for(let e=Math.floor(x0);e<=Math.ceil(x1);e++)for(const k of [1,2,5]){const v=k*10**e;const l=Math.log10(v);if(l>=x0&&l<=x1)xt.push(v)}}else{for(let i=0;i<=5;i++)xt.push(x0+(x1-x0)*i/5)}
 for(const v of xt){const x=X(v);s+=`<line class="ax" x1="${x}" x2="${x}" y1="${mt}" y2="${H-mb}"/><text class="tick" x="${x}" y="${H-mb+14}" text-anchor="middle">${xfmt(v)}</text>`}
 for(let i=0;i<=4;i++){const yv=y0+(y1-y0)*i/4;const v=ylog?10**yv:yv;const y=Y(v);s+=`<line class="ax" x1="${ml}" x2="${W-mr}" y1="${y}" y2="${y}"/><text class="tick" x="${ml-6}" y="${y+4}" text-anchor="end">${yfmt(v)}</text>`}
 if(xlog&&x0<0&&x1>0){const x=X(1);s+=`<line x1="${x}" x2="${x}" y1="${mt}" y2="${H-mb}" stroke="var(--ref)" stroke-dasharray="4 3"/>`}
 s+=`<text x="${(ml+W-mr)/2}" y="${H-8}" text-anchor="middle" class="tick">${xl}</text><text transform="translate(14 ${(mt+H-mb)/2}) rotate(-90)" text-anchor="middle" class="tick">${yl}</text>`;
 for(const p of pts){const c=p.m?OWN[p.m.owner]:'var(--ref)';const x=X(p.x),y=Y(p.y);
  s+=`<circle cx="${x}" cy="${y}" r="5" fill="${c}" fill-opacity="0.85" stroke="var(--panel)"><title>${p.m?p.m.label:'원 LLM-A*'}: x=${xfmt(p.x)}, y=${yfmt(p.y)}</title></circle><text x="${x+7}" y="${y-6}">${p.m?p.m.label:'LLM-A*'}</text>`}
 return s+'</svg>';}
const refPt=(x,y)=>({m:null,x,y});
document.getElementById('scatters').innerHTML=[
 scatter('속도 ↔ 경로 품질',m=>m.time&&m.time.v,m=>m.lstar,'계획기 시간 (원 LLM-A* 대비, 로그)','L/L* 기하평균 (낮을수록 최단에 가까움)',{ref:refPt(1,B.lstar),yfmt:v=>v.toFixed(3)}),
 scatter('속도 ↔ 최악 경로',m=>m.time&&m.time.v,m=>m.mx,'계획기 시간 (원 LLM-A* 대비, 로그)','최대 초과율 %',{ref:refPt(1,B.mx),yfmt:v=>v.toFixed(0)+'%'}),
 scatter('연산량 ↔ 시간',m=>m.exp&&m.exp.v,m=>m.time&&m.time.v,'이웃 확장 (원 LLM-A* 대비, 로그)','계획기 시간 (원 LLM-A* 대비, 로그)',{ylog:true,yfmt:v=>(100*v).toFixed(0)+'%',ref:refPt(1,1)}),
 scatter('총응답시간 ↔ LLM 응답 수',m=>m.e2e&&m.e2e.v,m=>m.resp,'총응답시간 추정 (원 LLM-A* 대비, 로그)','논리 LLM 응답 수 (1,000문제)',{yfmt:v=>Math.round(v).toLocaleString(),ref:refPt(1,B.resp)})
].join('');
// detail tabs
const GROUPS={
 '1 속도':[['계획기 시간 비율',m=>PCI(m.time)],['기하SD',m=>m.time?'×'+f(m.time.gsd,2):'—'],['개선/동일/악화',m=>m.time?`${m.time.dirs.improved}/${m.time.dirs.equal}/${m.time.dirs.worsened}`:'—'],['계획기 ms',m=>MS(m.ms,1000)],['총응답시간 비율',m=>PCI(m.e2e)],['총응답 s',m=>MS(m.e2e_s)],['반복 CV',m=>f(100*m.cv)+'%']],
 '2 연산량':[['이웃 확장 비율',m=>PCI(m.exp)],['확장 수',m=>MS(m.exp_n,1,1)],['Operation(논문) 비율',m=>P(m.op)],['heap pop 비율',m=>P(m.pops)],['Storage(논문) 비율',m=>P(m.stor)],['저장 상태 수',m=>MS(m.stor_n,1,1)]],
 '3 경로 품질':[['보장',m=>BND(m.bound)],['최단/유효',m=>`${m.opt}/${m.qn}`],['L/L* 기하평균',m=>f(m.lstar,4)],['L/L* SD',m=>f(m.lsd,4)],['P95 초과',m=>f(m.p95,2)+'%'],['최대 초과',m=>f(m.mx,2)+'%'],['5%↑',m=>m.o5],['10%↑',m=>m.o10],['길이 vs LLM-A*',m=>P(m.len)]],
 '4 LLM 비용':[['질의 시 LLM',m=>m.online?'예':'아니오'],['논리 응답',m=>Math.round(m.resp).toLocaleString()],['입력 토큰',m=>Math.round(m.tin).toLocaleString()],['출력 토큰',m=>Math.round(m.tout).toLocaleString()],['목록가 $',m=>f(m.usd,3)],['API 대기 s/문제',m=>MS(m.api)]],
 '5 메모리':[['OPEN 최대 비율',m=>PCI(m.open)],['OPEN 최대',m=>MS(m.open_n,1,1)],['Storage 비율',m=>P(m.stor)],['저장 상태 수',m=>MS(m.stor_n,1,1)]],
 '6 정확성':[['유효 경로',m=>`${m.valid}/${m.cases}`],['상한 위반',m=>m.viol],['반복 결정성',m=>'5회 동일']],
 '7 LLM 기여':[['대조군',m=>m.ctrl],['시간 비율',m=>PCI(m.ctl.planner_s)],['확장 비율',m=>P(m.ctl.expansions)],['길이 비율',m=>P(m.ctl.length)],['비고',m=>m.online?'LLM 안내 효과':'LLM 없음: 학습·제안 요소 효과']]};
const NOTES={'1 속도':'총응답시간 = 저장된 실제 API 대기 + 계획기 시간(재구성 추정). 반복 CV = 5회 시간의 변동계수 중앙값.','2 연산량':'엔진마다 OPEN 구현이 달라 heap pop은 직접 비교가 어렵다. 엔진 간 비교는 이웃 확장 수를 우선 본다.',
 '3 경로 품질':'S4·K5는 3시드 평균(품질 꼬리는 3,000개 시드별 경로).','4 LLM 비용':'gpt-4o-mini 목록가($0.15/$0.60 per 1M). S6의 LLM은 오프라인 계수 생성.','5 메모리':'논리 지표(OPEN 최대, 저장 상태 수). 프로세스 RSS는 이 블록에서 측정하지 않았다.',
 '6 정확성':'SM·JY 방법은 각자의 기록과 결정 필드 동등성을 확인한 뒤 측정했다(S4만 4,000회 중 14회 확장 수 1~3 차이, 경로 동일).','7 LLM 기여':'같은 엔진에서 LLM 안내를 빼거나 규칙·무작위로 바꾼 조건 대비 비율. 100% 미만이면 LLM 요소가 도움이 된 것.'};
const tabs=document.getElementById('tabs');let cur='1 속도';
function draw(){const cols=GROUPS[cur];let h='<tr><th class="l">방법</th>'+cols.map(c=>`<th>${c[0]}</th>`).join('')+'</tr>';
 for(const m of D.main) h+='<tr>'+nameCell(m)+cols.map(c=>`<td>${c[1](m)}</td>`).join('')+'</tr>';
 document.getElementById('detail').innerHTML=h;document.getElementById('dnote').textContent=NOTES[cur];
 for(const b of tabs.children) b.setAttribute('aria-pressed',b.textContent===cur);}
for(const k in GROUPS){const b=document.createElement('button');b.type='button';b.textContent=k;b.onclick=()=>{cur=k;draw()};tabs.appendChild(b)}
draw();
</script></body></html>
'''

if __name__ == '__main__':
    build()
