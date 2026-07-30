#!/usr/bin/env python3
"""
Behavioural similarity between methods (Table V, Fig 7).
Correlates each method's per-seed accuracy against every other method's,
across the 100 random partitions of each combination.

Run from the folder containing final_results/
"""
import csv, glob, os, re
import numpy as np
from scipy import stats

METHODS = ['gpfn','pcr','gblup','bayesb']
LAB = {'gpfn':'GPFN','gblup':'GBLUP','pcr':'PCR','bayesb':'BayesB'}
z  = lambda r: np.arctanh(np.clip(r, -0.999999, 0.999999))
iz = lambda x: np.tanh(x)

combos = {}
for f in sorted(glob.glob("final_results/final_*_2012_*.csv")):
    m = re.match(r'final_([A-Za-z]+)_2012_(yield|protein|oil)\.csv$', os.path.basename(f))
    rows = list(csv.DictReader(open(f)))
    combos[(m.group(1), m.group(2))] = {
        k: np.array([float(r[f'{k}_pearson']) for r in rows]) for k in METHODS}
KEYS = list(combos)
R = {k: {(a,b): np.corrcoef(c[a], c[b])[0,1] for a in METHODS for b in METHODS}
     for k, c in combos.items()}

print("TABLE V: behavioural similarity (Fisher-z averaged across 17 combos)")
pairs = [('gpfn','pcr'),('gblup','pcr'),('gpfn','gblup'),
         ('gblup','bayesb'),('pcr','bayesb'),('gpfn','bayesb')]
for a,b in pairs:
    zs = np.array([z(R[k][(a,b)]) for k in KEYS])
    se = zs.std(ddof=1)/np.sqrt(len(zs))
    raw = [R[k][(a,b)] for k in KEYS]
    print(f"  {LAB[a]+' ~ '+LAB[b]:20s} r={iz(zs.mean()):.3f}  "
          f"95% CI [{iz(zs.mean()-2.12*se):.3f}, {iz(zs.mean()+2.12*se):.3f}]  "
          f"range {min(raw):.3f} to {max(raw):.3f}")

def steiger(r_jk, r_jh, r_kh, n):
    """Steiger (1980): test two DEPENDENT correlations sharing variable j."""
    det = 1 - r_jk**2 - r_jh**2 - r_kh**2 + 2*r_jk*r_jh*r_kh
    if det <= 0: return np.nan, np.nan
    rbar = (r_jk + r_jh) / 2.0
    num = (r_jk - r_jh) * np.sqrt((n-1) * (1 + r_kh))
    den = np.sqrt(2*(n-1)/(n-3)*det + rbar**2 * (1-r_kh)**3)
    t = num/den
    return t, 2*(1 - stats.t.cdf(abs(t), n-3))

print("\nKey contrast: is GPFN more like PCR than like BayesB?")
ps = []
for k in KEYS:
    t,p = steiger(R[k][('gpfn','pcr')], R[k][('gpfn','bayesb')], R[k][('pcr','bayesb')], 100)
    ps.append(p)
n = len(ps); idx = np.argsort(ps); holm = np.empty(n); run = 0.
for rk,i in enumerate(idx):
    run = max(run, (n-rk)*ps[i]); holm[i] = min(run, 1.0)
print(f"  Steiger's test significant (uncorrected): {sum(1 for p in ps if p<0.05)}/17")
print(f"  Significant after Holm-Bonferroni:        {int((holm<0.05).sum())}/17")

zp = np.array([z(R[k][('gpfn','pcr')])    for k in KEYS])
zb = np.array([z(R[k][('gpfn','bayesb')]) for k in KEYS])
t,p   = stats.ttest_rel(zp, zb)
w,pw  = stats.wilcoxon(zp, zb)
d = zp - zb
print(f"  Paired t on Fisher z: t({len(KEYS)-1})={t:.2f}, p={p:.2e}, d={d.mean()/d.std(ddof=1):.2f}")
print(f"  Wilcoxon signed-rank: p={pw:.2e}")

rng = np.random.default_rng(42)
boots = np.array([iz(zp[s].mean()) - iz(zb[s].mean())
                  for s in rng.integers(0, 17, (10000, 17))])
print(f"  Difference in mean r: {iz(zp.mean())-iz(zb.mean()):+.3f}  "
      f"bootstrap 95% CI [{np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}]")
