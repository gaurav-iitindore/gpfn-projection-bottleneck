#!/usr/bin/env python3
"""
Regenerate every number reported in the manuscript from the raw per-partition
results, with no intermediate summary files.

    python analysis/reproduce_all_tables.py

Reads only:
    soybean/results/final_results/final_*.csv   (17 files, 100 rows each)
    soybean/results/heritability.csv
    barley/results/raw_results/py_*.csv         (9 files, 100 rows each)
    barley/results/raw_results/r_*.csv          (9 files, 100 rows each)

Prints Tables I to VI and every inline statistic, in manuscript order.
Every value printed here should match the submitted PDF exactly.
"""

import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOY = os.path.join(ROOT, "soybean")
BAR = os.path.join(ROOT, "barley")

METHODS = ["gpfn", "gblup", "pcr", "bayesb"]
LABEL = {"gpfn": "GPFN", "gblup": "GBLUP", "pcr": "PCR", "bayesb": "BayesB"}

z = lambda r: np.arctanh(np.clip(r, -0.999999, 0.999999))


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, ix in enumerate(order):
        val = (m - i) * p[ix]
        running = max(running, val)
        adj[ix] = min(running, 1.0)
    return adj


def steiger_z(r12, r13, r23, n):
    """Steiger (1980) test for two dependent correlations sharing one variable."""
    rm2 = (r12 ** 2 + r13 ** 2) / 2.0
    f = (1 - r23) / (2 * (1 - rm2))
    h = (1 - f * rm2) / (1 - rm2)
    return (z(r12) - z(r13)) * np.sqrt((n - 3) / (2 * (1 - r23) * h))


# ----------------------------------------------------------------------------
# Load soybean
# ----------------------------------------------------------------------------
soy_files = sorted(glob.glob(os.path.join(SOY, "results", "final_results", "final_*.csv")))
if not soy_files:
    raise SystemExit("No soybean result files found. Run from the repository root.")

per = {}
for f in soy_files:
    combo = os.path.basename(f).replace("final_", "").replace(".csv", "").replace("_2012", "")
    d = pd.read_csv(f)
    per[combo] = {m: d[m + "_pearson"].values for m in METHODS}
    per[combo]["n_train"] = d["n_train"].values

H = pd.read_csv(os.path.join(SOY, "results", "heritability.csv"))
h2 = {f"{r.location}_{r.trait}": r.h2 for r in H.itertuples()}

pooled = {m: np.concatenate([per[c][m] for c in per]) for m in METHODS}
N_PAIRED = len(pooled["gpfn"])

print(f"soybean: {len(per)} combinations x 100 partitions = {N_PAIRED} paired evaluations")
print(f"         {N_PAIRED * 4} model fits")

# ----------------------------------------------------------------------------
rule("TABLE I  Overall ranking across 1700 evaluations")
stack = np.vstack([pooled[m] for m in METHODS])
four_way = np.argmax(stack, axis=0)
best_in = dict.fromkeys(METHODS, 0)
for c in per:
    means = [per[c][m].mean() for m in METHODS]
    best_in[METHODS[int(np.argmax(means))]] += 1

order = sorted(METHODS, key=lambda m: -pooled[m].mean())
print(f"{'Rank':<5}{'Method':<9}{'Mean r':>9}{'SD':>8}{'Best in':>10}{'Partitions won':>16}")
for rank, m in enumerate(order, 1):
    won = (four_way == METHODS.index(m)).sum()
    print(f"{rank:<5}{LABEL[m]:<9}{pooled[m].mean():>9.3f}{pooled[m].std(ddof=1):>8.3f}"
          f"{f'{best_in[m]}/17':>10}{f'{won}/{N_PAIRED}':>16}")
print(f"\nNote: 'Partitions won' is a FOUR-WAY count (highest of all four methods)."
      f"\n      It is not a head-to-head count. See the head-to-head block below.")

# ----------------------------------------------------------------------------
rule("HEAD-TO-HEAD pairwise win counts (each row sums to 1700)")
for a, b in [("bayesb", "gpfn"), ("gpfn", "pcr"), ("gpfn", "gblup"),
             ("gblup", "bayesb"), ("pcr", "bayesb"), ("gblup", "pcr")]:
    wa = int((pooled[a] > pooled[b]).sum())
    wb = int((pooled[b] > pooled[a]).sum())
    print(f"  {LABEL[a]:>7} {wa:>5}   vs   {LABEL[b]:<7} {wb:<5}   ties {N_PAIRED - wa - wb}")
print("\n  Manuscript Sec. III-A quotes BayesB 1289 vs GPFN 411.")
print("  Manuscript Sec. III-B quotes GPFN 742 vs PCR.")

# ----------------------------------------------------------------------------
rule("TABLE II  Pairwise comparisons, pooled over 1700 paired evaluations")
print(f"{'Comparison':<20}{'Mean diff.':>11}{'95% CI':>24}{'d':>8}{'p':>12}")
for a, b in [("gpfn", "gblup"), ("gpfn", "pcr"), ("gpfn", "bayesb"),
             ("gblup", "pcr"), ("gblup", "bayesb"), ("pcr", "bayesb")]:
    diff = pooled[a] - pooled[b]
    _, p = stats.ttest_rel(pooled[a], pooled[b])
    d = diff.mean() / diff.std(ddof=1)
    se = diff.std(ddof=1) / np.sqrt(N_PAIRED)
    lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
    print(f"{LABEL[a] + ' - ' + LABEL[b]:<20}{diff.mean():>+11.3f}"
          f"{f'({lo:+.3f}, {hi:+.3f})':>24}{d:>+8.2f}{p:>12.1e}")

# ----------------------------------------------------------------------------
rule("TABLE III  BayesB versus GPFN by combination")
rows = []
for c in per:
    g, b = per[c]["gpfn"], per[c]["bayesb"]
    diff = b - g
    _, p = stats.ttest_rel(b, g)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    rows.append(dict(combo=c, N=round(per[c]["n_train"].mean()), h2=h2[c],
                     gpfn=g.mean(), bayesb=b.mean(), delta=diff.mean(),
                     lo=diff.mean() - 1.96 * se, hi=diff.mean() + 1.96 * se,
                     d=diff.mean() / diff.std(ddof=1), p_raw=p,
                     g_minus_pcr=(g - per[c]["pcr"]).mean()))
T = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
T["p_holm"] = holm(T.p_raw.values)

print(f"{'Location':<9}{'Trait':<9}{'N':>6}{'h2':>7}{'GPFN':>8}{'BayesB':>8}"
      f"{'Delta':>8}{'95% CI':>20}{'d':>7}{'p (Holm)':>11}")
for r in T.itertuples():
    loc, trait = r.combo.rsplit("_", 1)
    ph = "n.s." if r.p_holm >= 0.05 else f"{r.p_holm:.1e}"
    print(f"{loc:<9}{trait:<9}{r.N:>6}{r.h2:>7.3f}{r.gpfn:>8.3f}{r.bayesb:>8.3f}"
          f"{r.delta:>+8.3f}{f'({r.lo:+.3f}, {r.hi:+.3f})':>20}{r.d:>7.2f}{ph:>11}")
print(f"\n  BayesB > GPFN in {(T.delta > 0).sum()} of {len(T)} combinations")
print(f"  Significant after Holm-Bonferroni: {(T.p_holm < 0.05).sum()} of {len(T)}")
print(f"  Not significant: {', '.join(T.loc[T.p_holm >= 0.05, 'combo'])}")

# ----------------------------------------------------------------------------
rule("TABLE IV  Genomic heritability by trait")
agg = H.groupby("trait").h2.agg(["count", "mean", "min", "max"])
print(f"{'Trait':<10}{'Combinations':>14}{'Mean h2':>10}{'Range':>18}")
for t, r in agg.iterrows():
    print(f"{t:<10}{int(r['count']):>14}{r['mean']:>10.3f}{f'{r['min']:.3f}-{r['max']:.3f}':>18}")

# ----------------------------------------------------------------------------
rule("Sec. III-C and III-D  heritability relationships")
for lab, y in [("BayesB - GPFN gap vs h2", T.delta), ("GPFN - PCR gap vs h2", T.g_minus_pcr)]:
    r, pr = stats.pearsonr(T.h2, y)
    rho, ps = stats.spearmanr(T.h2, y)
    print(f"  {lab:<26} r = {r:+.3f} (p = {pr:.3f})   rho = {rho:+.3f} (p = {ps:.3f})")

rule("Sec. III-G  deficit versus training population size")
for comp in ["gblup", "bayesb"]:
    y = np.array([per[c]["gpfn"].mean() - per[c][comp].mean() for c in T.combo])
    r, p = stats.pearsonr(T.N.values, y)
    print(f"  GPFN - {LABEL[comp]:<7} vs N:  r = {r:+.3f} (p = {p:.4f})")

# ----------------------------------------------------------------------------
rule("TABLE V  Behavioural similarity, Fisher z-averaged across 17 combinations")
pair_r = {}
for a, b in combinations(METHODS, 2):
    rs = np.array([stats.pearsonr(per[c][a], per[c][b])[0] for c in per])
    pair_r[(a, b)] = rs
    mz = z(rs).mean()
    se = z(rs).std(ddof=1) / np.sqrt(len(rs))
    print(f"  {LABEL[a]:>7} ~ {LABEL[b]:<7}  mean r = {np.tanh(mz):.3f}   "
          f"95% CI ({np.tanh(mz - 1.96 * se):.3f}, {np.tanh(mz + 1.96 * se):.3f})   "
          f"range {rs.min():.3f} to {rs.max():.3f}")

gp, gb = pair_r[("gpfn", "pcr")], pair_r[("gpfn", "bayesb")]
print("\n  Key contrast: GPFN~PCR versus GPFN~BayesB")
print(f"    difference in mean r      {np.tanh(z(gp).mean()) - np.tanh(z(gb).mean()):+.3f}")
rng = np.random.default_rng(0)
bs = [np.tanh(z(gp[i]).mean()) - np.tanh(z(gb[i]).mean())
      for i in rng.integers(0, len(gp), (20000, len(gp)))]
print(f"    bootstrap 95% CI          ({np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f})")
t, p = stats.ttest_rel(z(gp), z(gb))
dz = (z(gp) - z(gb)).mean() / (z(gp) - z(gb)).std(ddof=1)
print(f"    paired t on Fisher z      t({len(gp) - 1}) = {t:.2f}, p = {p:.1e}")
print(f"    paired Cohen's d          {dz:.2f}")
print(f"    Wilcoxon signed-rank      p = {stats.wilcoxon(z(gp), z(gb))[1]:.1e}")

ps = []
for c in per:
    r12 = stats.pearsonr(per[c]["gpfn"], per[c]["pcr"])[0]
    r13 = stats.pearsonr(per[c]["gpfn"], per[c]["bayesb"])[0]
    r23 = stats.pearsonr(per[c]["pcr"], per[c]["bayesb"])[0]
    ps.append(2 * (1 - stats.norm.cdf(abs(steiger_z(r12, r13, r23, 100)))))
ps = np.array(ps)
print(f"    Steiger, per combination  significant in {(ps < 0.05).sum()}/{len(ps)}")
print(f"    after Holm-Bonferroni     significant in {(holm(ps) < 0.05).sum()}/{len(ps)}")

# ----------------------------------------------------------------------------
# Barley
# ----------------------------------------------------------------------------
rule("TABLE VI  Barley HEB-25, BayesB versus GPFN by target")
bar_files = sorted(glob.glob(os.path.join(BAR, "results", "raw_results", "py_*.csv")))
brows, ball = [], {m: [] for m in METHODS}
for f in bar_files:
    t = os.path.basename(f)[3:-4]
    p_ = pd.read_csv(f)
    r_ = pd.read_csv(os.path.join(BAR, "results", "raw_results", f"r_{t}.csv"))
    assert (p_.seed.values == r_.seed.values).all(), f"seed mismatch in {t}"
    assert (p_.n_train.values == r_.n_train.values).all(), f"n_train mismatch in {t}"
    d = {"gpfn": p_.gpfn_pearson.values, "gblup": p_.gblup_pearson.values,
         "pcr": p_.pcr_pearson.values, "bayesb": r_.bayesb_pearson.values}
    for m in d:
        ball[m].append(d[m])
    diff = d["bayesb"] - d["gpfn"]
    _, pp = stats.ttest_rel(d["bayesb"], d["gpfn"])
    brows.append(dict(target=t, trait="FT" if t == "FT" else "TGW",
                      arch="oligogenic" if t == "FT" else "polygenic",
                      N=round(p_.n_train.mean()),
                      gpfn=d["gpfn"].mean(), bayesb=d["bayesb"].mean(),
                      delta=diff.mean(), dz=diff.mean() / diff.std(ddof=1), p_raw=pp,
                      g_minus_pcr=(d["gpfn"] - d["pcr"]).mean()))
BT = pd.DataFrame(brows).sort_values(["arch", "N"], ascending=[False, True]).reset_index(drop=True)
BT["p_holm"] = holm(BT.p_raw.values)

print(f"{'Target':<16}{'Trait':<7}{'N':>6}{'GPFN':>8}{'BayesB':>8}{'Delta':>8}{'p (Holm)':>12}")
for r in BT.itertuples():
    ph = "n.s." if r.p_holm >= 0.05 else f"{r.p_holm:.1e}"
    print(f"{r.target:<16}{r.trait:<7}{r.N:>6}{r.gpfn:>8.3f}{r.bayesb:>8.3f}{r.delta:>+8.3f}{ph:>12}")
print("\n  N is the MEAN training size across the 100 partitions, computed here from")
print("  the raw files. Do not use the n_train column in summary_4method.csv, which")
print("  records a single partition and is 14 to 19 individuals too high.")

for m in ball:
    ball[m] = np.concatenate(ball[m])
print("\n  Overall means across 9 targets x 100 seeds:")
for m in sorted(METHODS, key=lambda k: -ball[k].mean()):
    print(f"    {LABEL[m]:<8}{ball[m].mean():.4f}")

poly, olig = BT[BT.arch == "polygenic"], BT[BT.arch == "oligogenic"]
print(f"\n  polygenic (8 targets): mean BayesB - GPFN = {poly.delta.mean():+.4f}, "
      f"significant {int((poly.p_holm < 0.05).sum())}/8")
print(f"  oligogenic FT:         BayesB - GPFN = {olig.delta.iloc[0]:+.4f}, "
      f"dz = {olig.dz.iloc[0]:.2f}, p_holm = {olig.p_holm.iloc[0]:.1e}")
print(f"\n  GPFN - PCR: mean signed {BT.g_minus_pcr.mean():+.4f}, "
      f"max absolute {BT.g_minus_pcr.abs().max():.4f} (on "
      f"{BT.loc[BT.g_minus_pcr.abs().idxmax(), 'target']})")

# ----------------------------------------------------------------------------
rule("Barley behavioural similarity  (NOT in the manuscript, reported for completeness)")
bgp, bgb = [], []
for f in bar_files:
    t = os.path.basename(f)[3:-4]
    p_ = pd.read_csv(f)
    r_ = pd.read_csv(os.path.join(BAR, "results", "raw_results", f"r_{t}.csv"))
    bgp.append(stats.pearsonr(p_.gpfn_pearson, p_.pcr_pearson)[0])
    bgb.append(stats.pearsonr(p_.gpfn_pearson, r_.bayesb_pearson)[0])
bgp, bgb = np.array(bgp), np.array(bgb)
print(f"  GPFN ~ PCR     mean r = {np.tanh(z(bgp).mean()):.3f}")
print(f"  GPFN ~ BayesB  mean r = {np.tanh(z(bgb).mean()):.3f}")
t, p = stats.ttest_rel(z(bgp), z(bgb))
print(f"  paired t on Fisher z: t({len(bgp) - 1}) = {t:.2f}, p = {p:.1e}")
print(f"\n  The direction replicates but the magnitude does not: the soybean gap is")
print(f"  +0.695 and the barley gap is {np.tanh(z(bgp).mean()) - np.tanh(z(bgb).mean()):+.3f}.")
print(f"  On eight polygenic targets all four methods perform near-identically, so")
print(f"  every pair of methods correlates highly. This is why the analysis is")
print(f"  reported for soybean only in the manuscript.")

print("\n" + "=" * 78)
print("Done. Every value above should match the submitted manuscript.")
print("=" * 78)
