"""Publication-style static plots from the saved experiment statistics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from core import ROOT

out=ROOT/'results/candidates_v1/report'
data=json.loads((out/'summary.json').read_text(encoding='utf-8'))
methods=['FOCAL_EUCLIDEAN','CONFORMAL','CHOMP_STANDARD','FMM_UNIFORM','FMM_LLM','TRANSPATH_11','TRANSPATH_22','TRANSPATH_33']
labels=['FOCAL baseline','Conformal region','CHOMP-inspired','FMM (no LLM)','FMM + LLM','Learned / seed 11','Learned / seed 22','Learned / seed 33']
rr={r['method']:r for r in data['summary'] if r['stage']=='main'}
colors=['#6d8191','#3e6ecc','#bd7137','#3a9588','#176f64','#8e65aa','#8e65aa','#8e65aa']
fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
y=np.arange(len(methods)); values=np.array([rr[m]['expansions_ratio'] for m in methods])
low=np.array([rr[m]['expansions_ci_low'] for m in methods]); high=np.array([rr[m]['expansions_ci_high'] for m in methods])
axes[0].barh(y,values,color=colors,height=.65)
axes[0].errorbar(values,y,xerr=[values-low,high-values],fmt='none',ecolor='#233541',capsize=3)
axes[0].axvline(1,color='#324756',linestyle='--',linewidth=1)
axes[0].set_yticks(y,labels); axes[0].invert_yaxis(); axes[0].set_xlabel('Expansion ratio vs FOCAL (map-bootstrap 95% CI)')
axes[0].set_title('Search effort | 200 unseen-map queries')
for i,v in enumerate(values): axes[0].text(v+.02,i,f'{v:.2f}',va='center',fontsize=9)
prep=np.array([rr[m]['preparation_ms_mean'] for m in methods]); search=np.array([rr[m]['search_ms_mean'] for m in methods]); total=np.array([rr[m]['local_cold_ms_mean'] for m in methods]);graph=total-prep-search
axes[1].barh(y,graph,label='Graph',color='#b7c5d0',height=.65)
axes[1].barh(y,prep,left=graph,label='Preparation / inference',color='#dea66d',height=.65)
axes[1].barh(y,search,left=graph+prep,label='Search',color='#477d98',height=.65)
axes[1].set_yticks(y,labels);axes[1].invert_yaxis();axes[1].set_xlabel('Mean local time per query (ms), excluding historical GPT calls')
axes[1].set_title('Preparation included | warmed resident neural model');axes[1].legend(loc='lower right',fontsize=8)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for seed,value in data['training'].items():
    h=value['history'];ax.plot([r['epoch'] for r in h],[r['validation_loss'] for r in h],label=f'Seed {seed} (best epoch {value["best_epoch"]})')
ax.set(xlabel='Epoch',ylabel='Weighted validation BCE',title='Training monitored only on separate validation maps');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
print(out/'comparison.png')
