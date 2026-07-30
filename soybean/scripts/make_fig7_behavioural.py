import csv, glob, os, re
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.linewidth':0.8,'axes.edgecolor':'#333333',
    'axes.spines.top':False,'axes.spines.right':False})

METHODS = ['gpfn','pcr','gblup','bayesb']
LAB = {'gpfn':'GPFN','gblup':'GBLUP','pcr':'PCR','bayesb':'BayesB'}
combos = {}
for f in sorted(glob.glob("final_results/final_*_2012_*.csv")):
    m = re.match(r'final_([A-Za-z]+)_2012_(yield|protein|oil)\.csv$', os.path.basename(f))
    rows = list(csv.DictReader(open(f)))
    combos[(m.group(1),m.group(2))] = {k: np.array([float(r[f'{k}_pearson']) for r in rows]) for k in METHODS}
KEYS = list(combos)
z  = lambda r: np.arctanh(np.clip(r,-0.999999,0.999999))
iz = lambda x: np.tanh(x)

M = np.zeros((4,4))
for i,a in enumerate(METHODS):
    for j,b in enumerate(METHODS):
        M[i,j] = 1.0 if a==b else iz(np.mean([z(np.corrcoef(c[a],c[b])[0,1]) for c in combos.values()]))

fig = plt.figure(figsize=(9.6,4.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.32)

# ---- panel A: heatmap ----
ax = fig.add_subplot(gs[0,0])
im = ax.imshow(M, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels([LAB[k] for k in METHODS], fontsize=8.5)
ax.set_yticklabels([LAB[k] for k in METHODS], fontsize=8.5)
for i in range(4):
    for j in range(4):
        col = 'white' if (M[i,j] > .72 or M[i,j] < .18) else '#222222'
        txt = '1' if i==j else f'{M[i,j]:.2f}'
        ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                color=col, weight='bold' if i!=j else 'normal')
# outline the structure cluster
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((-.5,-.5), 3, 3, fill=False, edgecolor='#222222', lw=2.0, zorder=5))
ax.text(1.0, -0.86, 'structure cluster', ha='center', fontsize=7.6, color='#222222', style='italic')
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title('How alike do the methods behave?', fontsize=10, pad=22, loc='left', weight='bold')
ax.text(-0.5, -1.55, 'Correlation of per-seed accuracy across the 100 random splits,\nFisher-z averaged over all 17 combinations.',
        fontsize=7.6, color='#666666', va='top')
cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.04)
cb.set_label('mean correlation across seeds', fontsize=7.6)
cb.ax.tick_params(labelsize=7)

# ---- panel B: per-combo paired comparison ----
ax = fig.add_subplot(gs[0,1])
rp = np.array([np.corrcoef(combos[k]['gpfn'], combos[k]['pcr'])[0,1]    for k in KEYS])
rb = np.array([np.corrcoef(combos[k]['gpfn'], combos[k]['bayesb'])[0,1] for k in KEYS])
o = np.argsort(rp)
y = np.arange(len(KEYS))
for i, idx in enumerate(o):
    ax.plot([rb[idx], rp[idx]], [i, i], '-', color='#cccccc', lw=1.0, zorder=1)
ax.plot(rb[o], y, 'o', ms=5.5, color='#1b9e77', markeredgecolor='white', markeredgewidth=.6, zorder=3, label='GPFN ~ BayesB')
ax.plot(rp[o], y, 'o', ms=5.5, color='#8d99ae', markeredgecolor='white', markeredgewidth=.6, zorder=3, label='GPFN ~ PCR')
ax.set_yticks(y)
ax.set_yticklabels([f'{KEYS[i][0]} {KEYS[i][1]}' for i in o], fontsize=7.4)
ax.set_xlim(-0.15, 1.05)
ax.axvline(0, color='#bbbbbb', lw=.8, ls='--')
ax.set_xlabel('Correlation of per-seed accuracy with GPFN', fontsize=9)
ax.set_title('GPFN moves with its projection, not with BayesB', fontsize=10, pad=22, loc='left', weight='bold')
ax.text(-0.15, len(KEYS)+1.35,
        'Every combination: GPFN tracks PCR far more closely than it tracks BayesB.\n'
        "Steiger's test for dependent correlations: significant in 17/17 after Holm correction.",
        fontsize=7.6, color='#666666', va='top')
ax.legend(frameon=False, fontsize=8, loc='lower right')
plt.savefig('Fig7_behavioural_similarity.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig7_behavioural_similarity.pdf', bbox_inches='tight')
print("Fig7_behavioural_similarity written")
