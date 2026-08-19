#!/usr/bin/env python3
"""
Fig 12: replacing the projection at inference does not recover the loss.
Regenerates in the paper's house style, matching scripts/make_figures.py exactly
(DejaVu Sans size 9, #333333 axes, spines off, 1.984*se CIs, Holm correction).
Sized to sit as a single-column figure consistent with the other forest plots.

    python make_fig12_projection.py --csv proj_IL_2012_protein_100.csv
Writes Fig12_projection_swap.pdf (and .png).
"""
import argparse, csv
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# EXACT house style from make_figures.py
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.linewidth':0.8,'axes.edgecolor':'#333333',
    'xtick.major.width':0.8,'ytick.major.width':0.8,
    'axes.spines.top':False,'axes.spines.right':False})

C = {'sig':'#1b9e77', 'ns':'#bbbbbb'}
ORDER = [("screen_pca:2000","Screen-then-PCA"),
         ("pls_mm","PLS, matched"),
         ("bag:10","Bagging"),
         ("pls","PLS"),
         ("hybrid:50","Hybrid")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='proj_IL_2012_protein_100.csv')
    ap.add_argument('--out', default='Fig12_projection_swap')
    a = ap.parse_args()

    by = defaultdict(dict)
    for r in csv.DictReader(open(a.csv)):
        by[int(r['seed'])][r['projection']] = float(r['gpfn_pearson'])
    seeds = sorted(by)

    names, deltas, los, his, raw_p = [], [], [], [], []
    for tok, lab in ORDER:
        d = np.array([by[s][tok] - by[s]['pca'] for s in seeds])
        _, p = stats.ttest_rel([by[s][tok] for s in seeds], [by[s]['pca'] for s in seeds])
        se = d.std(ddof=1)/np.sqrt(len(d))
        names.append(lab); deltas.append(d.mean())
        los.append(d.mean()-1.984*se); his.append(d.mean()+1.984*se); raw_p.append(p)

    # Holm-Bonferroni across the five comparisons
    n = len(raw_p); idx = np.argsort(raw_p); holm = np.empty(n); run = 0.
    for rk, i in enumerate(idx):
        run = max(run, (n-rk)*raw_p[i]); holm[i] = min(run, 1.0)

    y = np.arange(len(names))[::-1]
    # Larger figure so 9pt text is proportional, matching the other forest plots
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axvline(0, color='#999999', lw=1, ls='--', zorder=1)
    for i, yi in enumerate(y):
        col = C['sig'] if holm[i] < 0.05 else C['ns']
        ax.plot([los[i], his[i]], [yi, yi], color=col, lw=1.6, solid_capstyle='round', zorder=2)
        ax.plot([deltas[i]], [yi], 'o', color=col, ms=6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel('Replacement $-$ released projection (Pearson $r$)')
    ax.text(-0.004, y[0]+0.28, 'released\nprojection', fontsize=8, color='#666666',
            ha='right', va='bottom')
    ax.margins(y=0.16)
    ax.set_ylim(-0.6, len(names)-0.2)
    plt.tight_layout()
    plt.savefig(a.out+'.pdf', bbox_inches='tight')
    plt.savefig(a.out+'.png', dpi=200, bbox_inches='tight')
    print('wrote', a.out+'.pdf')
    for lab, d, h in zip(names, deltas, holm):
        star = '***' if h<0.001 else '**' if h<0.01 else '*' if h<0.05 else 'ns'
        print(f'  {lab:16s} {d:+.4f}  Holm {star}')

if __name__ == '__main__':
    main()
