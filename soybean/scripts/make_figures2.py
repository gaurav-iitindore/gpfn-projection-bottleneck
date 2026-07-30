import csv, glob, os, re
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.linewidth':0.8,'axes.edgecolor':'#333333','axes.spines.top':False,'axes.spines.right':False})
h2 = {}
for r in csv.DictReader(open('heritability.csv')):
    h2[(r['location'], r['trait'])] = float(r['h2'])
combos = {}
for f in sorted(glob.glob("final_results/final_*_2012_*.csv")):
    m = re.match(r'final_([A-Za-z]+)_2012_(yield|protein|oil)\.csv$', os.path.basename(f))
    loc, tr = m.group(1), m.group(2)
    d = {k: [] for k in ['gpfn','gblup','pcr','bayesb']}; nt = None
    for r in csv.DictReader(open(f)):
        nt = int(r['n_train'])
        for k in d: d[k].append(float(r[f'{k}_pearson']))
    combos[(loc,tr)] = {'n': nt, **{k: np.array(v) for k,v in d.items()}}
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
mk = {'yield':'o','protein':'s','oil':'^'}
tcol = {'yield':'#c44e52','protein':'#4c72b0','oil':'#55a868'}
H  = np.array([h2[k] for k in combos])
BB = np.array([(c['bayesb']-c['gpfn']).mean() for c in combos.values()])
GP = np.array([(c['gpfn']-c['pcr']).mean() for c in combos.values()])
TR = [k[1] for k in combos]
xs = np.linspace(H.min()-.01, H.max()+.01, 50)
ax = axes[0]
for i, t in enumerate(TR):
    ax.plot(H[i], BB[i], mk[t], color=tcol[t], ms=7, alpha=.9, markeredgecolor='white', markeredgewidth=.7)
r, p = stats.pearsonr(H, BB); rs, ps = stats.spearmanr(H, BB)
sl, ic = np.polyfit(H, BB, 1)
ax.plot(xs, sl*xs+ic, '-', color='#444444', lw=1.2, alpha=.6)
ax.axhline(0, color='#999999', lw=.9, ls='--')
ax.set_xlabel('Genomic heritability $h^2$', fontsize=9.5)
ax.set_ylabel('BayesB $-$ GPFN  (Pearson $r$)', fontsize=9.5)
ax.set_title('BayesB gains most where $h^2$ is high\n$r$ = %+.3f ($p$ = %.3f),  $\\rho$ = %+.3f ($p$ = %.4f)' % (r,p,rs,ps), fontsize=9.5, pad=8)
ax = axes[1]
for i, t in enumerate(TR):
    ax.plot(H[i], GP[i], mk[t], color=tcol[t], ms=7, alpha=.9, markeredgecolor='white', markeredgewidth=.7)
r2, p2 = stats.pearsonr(H, GP); rs2, ps2 = stats.spearmanr(H, GP)
sl, ic = np.polyfit(H, GP, 1)
ax.plot(xs, sl*xs+ic, '-', color='#444444', lw=1.2, alpha=.6)
ax.axhline(0, color='#999999', lw=.9, ls='--')
ax.text(.98,.96,'GPFN better than its own control',transform=ax.transAxes,ha='right',va='top',fontsize=7.4,color='#666666')
ax.text(.98,.04,'GPFN worse than its own control',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.4,color='#666666')
ax.set_xlabel('Genomic heritability $h^2$', fontsize=9.5)
ax.set_ylabel('GPFN $-$ PCR  (Pearson $r$)', fontsize=9.5)
ax.set_title('GPFN falls below PCR as $h^2$ rises\n$r$ = %+.3f ($p$ = %.3f),  $\\rho$ = %+.3f ($p$ = %.3f)' % (r2,p2,rs2,ps2), fontsize=9.5, pad=8)
handles = [Line2D([],[],marker=mk[t], color=tcol[t], ls='', ms=6, label=t) for t in ['yield','protein','oil']]
axes[0].legend(handles=handles, frameon=False, fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig('fig4_heritability_mechanism.png', dpi=300, bbox_inches='tight')
plt.savefig('fig4_heritability_mechanism.pdf', bbox_inches='tight')
plt.close()
print("fig4 done")
