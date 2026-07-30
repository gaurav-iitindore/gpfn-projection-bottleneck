#!/usr/bin/env python3
import csv, glob, os, re
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.linewidth':0.8,'axes.edgecolor':'#333333',
    'xtick.major.width':0.8,'ytick.major.width':0.8,
    'axes.spines.top':False,'axes.spines.right':False})
C={'gpfn':'#d1495b','gblup':'#30638e','pcr':'#8d99ae','bayesb':'#1b9e77'}
LAB={'gpfn':'GPFN','gblup':'GBLUP','pcr':'PCR','bayesb':'BayesB'}
METHODS=['bayesb','gblup','pcr','gpfn']
combos={}
for f in sorted(glob.glob("final_results/final_*_2012_*.csv")):
    m=re.match(r'final_([A-Za-z]+)_2012_(yield|protein|oil)\.csv$', os.path.basename(f))
    loc,tr=m.group(1),m.group(2); d={k:[] for k in ['gpfn','gblup','pcr','bayesb']}; nt=None
    for r in csv.DictReader(open(f)):
        nt=int(r['n_train'])
        for k in d: d[k].append(float(r[f'{k}_pearson']))
    combos[(loc,tr)]={'n_train':nt, **{k:np.array(v) for k,v in d.items()}}
order=sorted(combos.items(), key=lambda x:x[1]['n_train'])

fig,ax=plt.subplots(figsize=(7.0,6.2))
names,deltas,los,his,sigs,raw_p=[],[],[],[],[],[]
for (loc,tr),c in order:
    d=c['bayesb']-c['gpfn']; t,p=stats.ttest_rel(c['bayesb'],c['gpfn'])
    se=d.std(ddof=1)/np.sqrt(len(d))
    names.append(f"{loc} {tr}  (N={c['n_train']})")
    deltas.append(d.mean()); los.append(d.mean()-1.984*se); his.append(d.mean()+1.984*se); raw_p.append(p)
n=len(raw_p); idx=np.argsort(raw_p); holm=np.empty(n); run=0.
for rk,i in enumerate(idx):
    run=max(run,(n-rk)*raw_p[i]); holm[i]=min(run,1.0)
for h in holm: sigs.append('***' if h<0.001 else '**' if h<0.01 else '*' if h<0.05 else 'ns')
y=np.arange(len(names)); ax.axvline(0,color='#999999',lw=1,ls='--',zorder=1)
for i in range(len(y)):
    col=C['bayesb'] if holm[i]<0.05 else '#bbbbbb'
    ax.plot([los[i],his[i]],[y[i],y[i]],color=col,lw=1.6,solid_capstyle='round',zorder=2)
    ax.plot(deltas[i],y[i],'o',color=col,ms=5.5,zorder=3)
    ax.text(his[i]+0.004,y[i],sigs[i],va='center',fontsize=7.5,color='#555555')
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=8); ax.invert_yaxis()
ax.set_xlabel('Accuracy difference:  BayesB $-$ GPFN  (Pearson $r$)',fontsize=9.5)
ax.set_title('BayesB outperforms GPFN in 16 of 17 location-trait combinations',fontsize=10.5,pad=22,loc='left',weight='bold')
ax.text(0,1.005,'Points are mean differences over 100 seeds; bars are 95% CI. Coloured = significant after Holm correction.',
        transform=ax.transAxes,fontsize=7.8,color='#666666',va='bottom')
ax.set_xlim(-0.03,0.155); plt.tight_layout()
plt.savefig('fig1_forest_bayesb_vs_gpfn.png',dpi=300,bbox_inches='tight')
plt.savefig('fig1_forest_bayesb_vs_gpfn.pdf',bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(8.4,4.6))
x=np.arange(len(order)); w=0.2
for j,k in enumerate(METHODS):
    means=np.array([c[k].mean() for _,c in order])
    sems=np.array([c[k].std(ddof=1)/np.sqrt(len(c[k])) for _,c in order])
    ax.bar(x+(j-1.5)*w,means,w,yerr=1.96*sems,label=LAB[k],color=C[k],
           edgecolor='white',linewidth=0.4,error_kw=dict(lw=0.7,capsize=1.6,ecolor='#555555'))
ax.axhline(0,color='#333333',lw=0.8); ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n{t}" for (l,t),_ in order],fontsize=7.2)
ax.set_ylabel('Prediction accuracy (Pearson $r$)',fontsize=9.5)
ax.set_title('Prediction accuracy across 17 SoyNAM location-trait combinations',fontsize=10.5,pad=26,loc='left',weight='bold')
ax.text(0,1.005,'Between-families prediction, 100 seeds per combination. Ordered by training population size.',
        transform=ax.transAxes,fontsize=7.8,color='#666666',va='bottom')
ax.legend(frameon=False,ncol=4,fontsize=8.5,loc='upper left',bbox_to_anchor=(0.0,0.98))
ax.set_ylim(-0.09,0.70); plt.tight_layout()
plt.savefig('fig2_accuracy_by_combo.png',dpi=300,bbox_inches='tight')
plt.savefig('fig2_accuracy_by_combo.pdf',bbox_inches='tight'); plt.close()

fig,axes=plt.subplots(1,2,figsize=(8.6,3.7),sharey=True)
N=np.array([c['n_train'] for _,c in order],float); mk={'yield':'o','protein':'s','oil':'^'}
for ax,comp in zip(axes,['gblup','bayesb']):
    gap=np.array([(c['gpfn']-c[comp]).mean() for _,c in order])
    for i,((l,t),c) in enumerate(order):
        ax.plot(N[i],gap[i],mk[t],color=C[comp],ms=6,alpha=.85,markeredgecolor='white',markeredgewidth=.6)
    r,p=stats.pearsonr(N,gap); sl,ic=np.polyfit(N,gap,1)
    xs=np.linspace(N.min(),N.max(),50); ax.plot(xs,sl*xs+ic,'-',color=C[comp],lw=1.3,alpha=.55)
    ax.axhline(0,color='#999999',lw=.9,ls='--'); ax.set_xlabel('Training population size',fontsize=9)
    ax.set_title(f'GPFN $-$ {LAB[comp]}\n$r$ = {r:+.3f},  $p$ = {p:.3f}',fontsize=9.5,pad=6)
axes[0].set_ylabel('GPFN accuracy deficit (Pearson $r$)',fontsize=9)
handles=[Line2D([],[],marker=mk[t],color='#555555',ls='',ms=5.5,label=t) for t in ['yield','protein','oil']]
axes[1].legend(handles=handles,frameon=False,fontsize=8,loc='lower left')
fig.suptitle('GPFN accuracy deficit widens with training population size',
             fontsize=10.5,x=.005,ha='left',weight='bold',y=1.12)
fig.text(.005,1.02,'Each point is one location-trait combination (n=17). Note training sizes are unevenly distributed, with 12 of 17 combinations near N=4000.',
         fontsize=7.8,color='#666666',ha='left')
plt.tight_layout()
plt.savefig('fig3_deficit_vs_n.png',dpi=300,bbox_inches='tight')
plt.savefig('fig3_deficit_vs_n.pdf',bbox_inches='tight'); plt.close()
print("wrote fig1, fig2, fig3 (png + pdf)")
