#!/usr/bin/env python3
"""
Genomic heritability per location-trait combination.
Builds the full-population GRM from the same marker matrix the methods see,
then estimates h2 by REML (average information / grid on the profile likelihood).
"""
import os, subprocess, sys, shutil, csv
import numpy as np
from joblib import load

COMBOS = [('MI','yield'),('MO','yield'),('OHmc','yield'),('OHmi','yield'),
          ('KS','yield'),('IA','yield'),('IL','yield'),('IN','yield'),('NE','yield'),
          ('IA','protein'),('IL','protein'),('IN','protein'),('NE','protein'),
          ('IA','oil'),('IL','oil'),('IN','oil'),('NE','oil')]
HAP = 'soynam_29416_imputed.hmp.txt'

def build_bin(loc, trait, workdir):
    """Run variants2bin on the FULL population for this combo (no split)."""
    os.makedirs(workdir, exist_ok=True)
    pheno = f'soynam_{loc}_2012_{trait}_filtered.tsv'
    shutil.copy(HAP, os.path.join(workdir, HAP))
    shutil.copy(pheno, os.path.join(workdir, pheno))
    # phenotype column is literally named 'yield' in every file
    cmd = ['python', os.path.abspath('parsing/variants2bin.py'),
           '--genotype_file', HAP, '--phenotype_file', pheno,
           '--phenotype_name', 'yield']
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    if r.returncode != 0:
        raise RuntimeError(f'{loc} {trait}: {r.stderr[-400:]}')
    return os.path.join(workdir, 'yield.variants2.bin')

def grm(X):
    """VanRaden GRM."""
    X = np.asarray(X, dtype=np.float64)
    p = X.mean(axis=0) / 2.0
    keep = (p > 0.01) & (p < 0.99)
    X = X[:, keep]; p = p[keep]
    Z = X - 2.0 * p
    denom = 2.0 * np.sum(p * (1.0 - p))
    return (Z @ Z.T) / denom

def reml_h2(G, y):
    """Profile-likelihood REML for h2 in y = mu + g + e,  var(g)=h2*s2*G, var(e)=(1-h2)*s2*I."""
    y = np.asarray(y, dtype=np.float64)
    y = (y - y.mean()) / y.std()
    n = len(y)
    lam, U = np.linalg.eigh(G)
    lam = np.clip(lam, 0, None)
    Uty = U.T @ y
    Ut1 = U.T @ np.ones(n)
    def negll(h2):
        d = h2 * lam + (1.0 - h2)
        d = np.clip(d, 1e-8, None)
        # GLS mean
        mu = (Ut1 / d @ Uty) / (Ut1 / d @ Ut1)
        r = Uty - mu * Ut1
        s2 = np.sum(r * r / d) / (n - 1)
        return 0.5 * (np.sum(np.log(d)) + (n - 1) * np.log(s2)
                      + np.log(np.sum(Ut1 * Ut1 / d)))
    grid = np.linspace(0.001, 0.999, 200)
    vals = np.array([negll(h) for h in grid])
    h0 = grid[int(np.argmin(vals))]
    # refine
    fine = np.linspace(max(0.001, h0 - 0.01), min(0.999, h0 + 0.01), 100)
    vals2 = np.array([negll(h) for h in fine])
    return float(fine[int(np.argmin(vals2))])

rows = []
for i, (loc, tr) in enumerate(COMBOS, 1):
    wd = f'h2_work/{loc}_{tr}'
    print(f'[{i:2d}/17] {loc:5s} {tr:8s} ...', end=' ', flush=True)
    try:
        b = build_bin(loc, tr, wd)
        d = load(b)
        X, y = np.array(d[0]), np.array(d[1], dtype=float)
        ok = ~np.isnan(y)
        X, y = X[ok], y[ok]
        G = grm(X)
        h2 = reml_h2(G, y)
        rows.append({'location': loc, 'trait': tr, 'n': int(len(y)),
                     'markers': int(X.shape[1]), 'h2': round(h2, 4)})
        print(f'n={len(y):5d}  h2={h2:.3f}', flush=True)
    except Exception as e:
        print('FAILED:', e, flush=True)
    finally:
        shutil.rmtree(wd, ignore_errors=True)

with open('heritability.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['location','trait','n','markers','h2'])
    w.writeheader(); w.writerows(rows)

print('\n=== genomic heritability by trait ===')
for tr in ['yield','protein','oil']:
    v = [r['h2'] for r in rows if r['trait'] == tr]
    if v:
        print(f'  {tr:8s}  mean h2 = {np.mean(v):.3f}   range {min(v):.3f} to {max(v):.3f}')
print('\nwrote heritability.csv')
