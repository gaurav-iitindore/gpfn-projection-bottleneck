#!/usr/bin/env python3
"""
Fig12: replacing the projection at inference does not recover the loss.
Reads the raw per-seed runner output and renders in the paper's house style,
matching scripts/make_figures.py (fonts, colours, spines, CI = 1.984*se, Holm).

Run where the CSV lives (.109 or PARAM), conda env gpfn:
    python make_fig12_projection.py --csv proj_IL_2012_protein_100.csv
Writes Fig12_projection_swap.pdf (and .png) in the current directory.
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.linewidth':0.8,'axes.edgecolor':'#333333',
    'xtick.major.width':0.8,'ytick.major.width':0.8,
    'axes.spines.top':False,'axes.spines.right':False})

# same palette as make_figures.py; green marks a significant difference
C = {'sig':'#1b9e77', 'ns':'#bbbbbb'}

# csv token -> display label, ordered best (closest to zero) first
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
        t, p = stats.ttest_rel([by[s][tok] for s in seeds], [by[s]['pca'] for s in seeds])
        se = d.std(ddof=1)/np.sqrt(len(d))
        names.append(lab); deltas.append(d.mean())
        los.append(d.mean()-1.984*se); his.append(d.mean()+1.984*se); raw_p.append(p)

    # Holm-Bonferroni across the five comparisons, exactly as make_figures.py
    n = len(raw_p); idx = np.argsort(raw_p); holm = np.empty(n); run = 0.
    for rk, i in enumerate(idx):
        run = max(run, (n-rk)*raw_p[i]); holm[i] = min(run, 1.0)

    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axvline(0, color='#999999', lw=1, ls='--', zorder=1)
    for i, yi in enumerate(y):
        col = C['sig'] if holm[i] < 0.05 else C['ns']
        ax.plot([los[i], his[i]], [yi, yi], color=col, lw=1.6, solid_capstyle='round', zorder=2)
        ax.plot([deltas[i]], [yi], 'o', color=col, ms=5, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel('Replacement $-$ released projection (Pearson $r$)')
    ax.text(0.001, y[0]+0.6, 'released\nprojection', fontsize=7, color='#666666',
            ha='left', va='bottom')
    ax.margins(y=0.18)
    plt.tight_layout()
    plt.savefig(a.out+'.pdf', bbox_inches='tight')
    plt.savefig(a.out+'.png', dpi=200, bbox_inches='tight')
    print('wrote', a.out+'.pdf')
    for lab, d, lo, hi, h in zip(names, deltas, los, his, holm):
        star = '***' if h<0.001 else '**' if h<0.01 else '*' if h<0.05 else 'ns'
        print(f'  {lab:16s} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  Holm {star}')

if __name__ == '__main__':
    main()
