#!/usr/bin/env python3
"""Merge BayesB into the existing GPFN/GBLUP/PCR results, by seed. Strict on split match."""
import csv, glob, os, re, statistics as st

os.makedirs("final_results", exist_ok=True)
SPLIT_KEYS = ['n_train','n_test','n_families_train','n_families_test']
summary = []

for old_f in sorted(glob.glob("old_results/results_*_2012_*.csv")):
    m = re.match(r'results_([A-Za-z]+)_2012_(yield|protein|oil)\.csv$', os.path.basename(old_f))
    if not m: continue
    loc, tr = m.group(1), m.group(2)
    new_f = f"bayesb_results/bayesb_{loc}_2012_{tr}.csv"
    if not os.path.exists(new_f):
        print(f"  SKIP {loc} {tr}: no BayesB file"); continue

    old = {int(r['seed']): r for r in csv.DictReader(open(old_f))}
    new = {int(r['seed']): r for r in csv.DictReader(open(new_f))}
    seeds = sorted(set(old) & set(new))

    # strict split check
    bad = [s for s in seeds if tuple(old[s][k] for k in SPLIT_KEYS) != tuple(new[s][k] for k in SPLIT_KEYS)]
    if bad:
        print(f"  REFUSED {loc} {tr}: {len(bad)} split mismatches, not merged"); continue

    out_f = f"final_results/final_{loc}_2012_{tr}.csv"
    fields = ['seed','n_train','n_test','n_families_train','n_families_test',
              'gpfn_pearson','gpfn_spearman','gblup_pearson','gblup_spearman',
              'pcr_pearson','pcr_spearman','bayesb_pearson','bayesb_spearman']
    with open(out_f,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for s in seeds:
            row = {k: old[s][k] for k in fields if k in old[s]}
            row['seed'] = s
            row['bayesb_pearson']  = new[s]['bayesb_pearson']
            row['bayesb_spearman'] = new[s]['bayesb_spearman']
            w.writerow(row)

    def mean(key, src):
        vals = [float(src[s][key]) for s in seeds
                if src[s].get(key) not in (None,'','nan')]
        return st.mean(vals) if vals else float('nan')
    summary.append({
        'combo': f"{loc}_{tr}", 'n_seeds': len(seeds),
        'n_train': old[seeds[0]]['n_train'],
        'gpfn':   mean('gpfn_pearson', old),
        'gblup':  mean('gblup_pearson', old),
        'pcr':    mean('pcr_pearson', old),
        'bayesb': mean('bayesb_pearson', new),
    })
    print(f"  merged {loc:5s} {tr:8s} ({len(seeds)} seeds) -> {out_f}")

# summary table
with open("final_results/summary_4method.csv","w",newline='') as f:
    w = csv.DictWriter(f, fieldnames=['combo','n_train','n_seeds','gpfn','gblup','pcr','bayesb'])
    w.writeheader()
    for r in sorted(summary, key=lambda x: int(x['n_train'])):
        w.writerow({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})

print("\n==== MEAN PEARSON r (sorted by training N) ====")
print(f"{'combo':16s} {'n_train':>7s} {'GPFN':>8s} {'GBLUP':>8s} {'PCR':>8s} {'BayesB':>8s}   best")
for r in sorted(summary, key=lambda x: int(x['n_train'])):
    vals = {'GPFN':r['gpfn'],'GBLUP':r['gblup'],'PCR':r['pcr'],'BayesB':r['bayesb']}
    best = max(vals, key=vals.get)
    print(f"{r['combo']:16s} {r['n_train']:>7s} {r['gpfn']:8.4f} {r['gblup']:8.4f} "
          f"{r['pcr']:8.4f} {r['bayesb']:8.4f}   {best}")
print(f"\nMerged {len(summary)} combos. Files in final_results/")
