"""
Statistical tests for the barley HEB-25 benchmark, matching the Paper 1
(SoyNAM) reporting standard.

For each target, methods are compared on the SAME 100 between-families
partitions (paired), so we use paired t-tests on the per-seed Pearson r.
The primary contrast is BayesB - GPFN (does the sparse-prior method beat the
transformer?). p-values are Holm-Bonferroni corrected across the 9 targets.

Outputs:
  results/statistical_tests.csv   , per-target table (Table III analogue)
  console                          , summary
"""
import csv, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(HERE, "results", "raw_results")

TARGETS = ["Dundee_2014_N0","Dundee_2014_N1","Dundee_2015_N0","Dundee_2015_N1",
           "Halle_2014_N0","Halle_2014_N1","Halle_2015_N0","Halle_2015_N1","FT"]
ARCH = {t: ("oligogenic" if t == "FT" else "polygenic") for t in TARGETS}
TRAIT = {t: ("flowering" if t == "FT" else "grain_weight") for t in TARGETS}


def read_paired(target):
    """Return dict of per-seed Pearson r arrays, aligned by seed."""
    py = os.path.join(RAW, f"py_{target}.csv")
    rr = os.path.join(RAW, f"r_{target}.csv")
    def load(path):
        d = {}
        for row in csv.DictReader(open(path)):
            d[int(row["seed"])] = row
        return d
    P, R = load(py), load(rr)
    seeds = sorted(set(P) & set(R))
    def arr(src, key):
        out = []
        for s in seeds:
            try: out.append(float(src[s][key]))
            except (ValueError, KeyError): out.append(np.nan)
        return np.array(out)
    return dict(
        seeds=seeds,
        n_train=int(float(P[seeds[0]]["n_train"])),
        gpfn=arr(P, "gpfn_pearson"),
        gblup=arr(P, "gblup_pearson"),
        pcr=arr(P, "pcr_pearson"),
        bayesb=arr(R, "bayesb_pearson"),
    )


def paired_stats(a, b):
    """Paired comparison a - b. Returns mean diff, 95% CI, Cohen's d, p (two-sided)."""
    d = a - b
    d = d[~np.isnan(d)]
    n = len(d)
    md = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (md - tcrit * se, md + tcrit * se)
    dz = md / sd  # Cohen's d for paired (dz)
    t, p = stats.ttest_rel(a, b, nan_policy="omit")
    return md, ci, dz, p


def holm(pvals):
    """Holm-Bonferroni corrected p-values, preserving input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        corrected[idx] = min(running, 1.0)
    return corrected


def main():
    rows = []
    raw_p = []
    for t in TARGETS:
        d = read_paired(t)
        md, ci, dz, p = paired_stats(d["bayesb"], d["gpfn"])
        rows.append(dict(
            target=t, trait=TRAIT[t], architecture=ARCH[t], n_train=d["n_train"],
            gpfn=round(np.nanmean(d["gpfn"]), 4),
            gblup=round(np.nanmean(d["gblup"]), 4),
            pcr=round(np.nanmean(d["pcr"]), 4),
            bayesb=round(np.nanmean(d["bayesb"]), 4),
            bayesb_minus_gpfn=round(md, 4),
            ci_low=round(ci[0], 4), ci_high=round(ci[1], 4),
            cohens_dz=round(dz, 3),
            p_raw=p,
        ))
        raw_p.append(p)

    p_holm = holm(np.array(raw_p))
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = ph
        r["significant_holm"] = "yes" if ph < 0.05 else "no"

    # write CSV
    out = os.path.join(HERE, "results", "statistical_tests.csv")
    fields = ["target","trait","architecture","n_train","gpfn","gblup","pcr","bayesb",
              "bayesb_minus_gpfn","ci_low","ci_high","cohens_dz","p_raw","p_holm","significant_holm"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["p_raw"] = f"{r['p_raw']:.2e}"
            rr["p_holm"] = f"{r['p_holm']:.2e}"
            w.writerow(rr)

    # console summary
    print("=== BayesB vs GPFN, paired per-target (Holm-corrected across 9 targets) ===\n")
    print(f"{'Target':16s} {'arch':11s} {'N':>5s} {'B-G':>7s} {'95% CI':>18s} {'dz':>6s} {'p_holm':>9s} {'sig':>4s}")
    for r in rows:
        ci = f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
        print(f"{r['target']:16s} {r['architecture']:11s} {r['n_train']:5d} "
              f"{r['bayesb_minus_gpfn']:+7.4f} {ci:>18s} {r['cohens_dz']:+6.2f} "
              f"{r['p_holm']:9.2e} {r['significant_holm']:>4s}")

    n_sig = sum(1 for r in rows if r["significant_holm"] == "yes")
    n_bwin = sum(1 for r in rows if r["bayesb_minus_gpfn"] > 0)
    print(f"\nBayesB > GPFN (point estimate) in {n_bwin}/9 targets")
    print(f"Significant after Holm-Bonferroni in {n_sig}/9 targets")

    poly = [r for r in rows if r["architecture"] == "polygenic"]
    olig = [r for r in rows if r["architecture"] == "oligogenic"]
    print(f"\nPolygenic (grain weight): mean BayesB-GPFN = "
          f"{np.mean([r['bayesb_minus_gpfn'] for r in poly]):+.4f}, "
          f"significant in {sum(1 for r in poly if r['significant_holm']=='yes')}/{len(poly)}")
    print(f"Oligogenic (flowering):   BayesB-GPFN = "
          f"{olig[0]['bayesb_minus_gpfn']:+.4f}, "
          f"significant: {olig[0]['significant_holm']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
