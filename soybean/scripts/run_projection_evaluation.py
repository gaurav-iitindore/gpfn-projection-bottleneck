#!/usr/bin/env python3
"""
run_projection_evaluation.py

Option A runner. For each between-families seed and each requested PROJECTION,
evaluate GPFN and a PCR control on BYTE-IDENTICAL features, and log one CSV row
per (seed, projection).

It reuses the exact split and binary pipeline the frozen results used
(split_between_families.py -> parsing/variants2bin.py), so with projection 'pca'
the GPFN column reproduces the frozen per-seed numbers. The only thing that
varies across projections is the 100-column map fed to the transformer; GPFN and
PCR always read the same array, which is the control that separates "the features
are better" from "GPFN cannot use better features".

GBLUP and BayesB are projection-independent (both read raw markers), so this
runner does NOT recompute them. Merge the frozen per-seed GBLUP/BayesB columns by
seed afterwards.

Projections (each keeps the frozen 100 slots):
    pca                 stock PCA. V0 baseline, reproduces frozen GPFN.
    pls                 PLS, 100 components.
    pls_mm              PLS, moment-matched to the PCA variance profile.
    screen_pca:K        top-K markers by training association, then PCA to 100.
    hybrid:P            P principal components + (100-P) screened markers.
    bag:B               B disjoint marker subsets, PCA each, average GPFN preds.

Example (run on AgriHub, in ~/gpfn, conda env gpfn active, GPU visible, tmux):

    python run_projection_evaluation.py \
        --hapmap soynam_29416_imputed.hmp.txt \
        --pheno  soynam_IL_2012_protein_filtered.tsv \
        --phenotype_name protein \
        --model_path deploy/pika.pt \
        --projections pca,pls,pls_mm,screen_pca:2000,hybrid:50,bag:10 \
        --n_repeats 100 \
        --out_csv proj_IL_2012_protein.csv \
        --work_dir proj_runs_IL_2012_protein

Output CSV columns (one row per seed per projection):
    seed, projection, n_train, n_test, n_families_train, n_families_test,
    gpfn_pearson, gpfn_spearman, pcr_pearson, pcr_spearman, eff_rank,
    wall_clock_seconds
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from joblib import load
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

import torch
from util.tools import normalize, center_markers, load_model
from proj_ext import feature_reduce_ext, bagged_feature_sets, effective_rank, gpfn_predict_from_features


def _spearman_stat(result):
    return getattr(result, "statistic", getattr(result, "correlation", None))


# ---- split + binary pipeline, lifted from run_repeated_evaluation.py ---------
def run_split(hapmap, pheno, out_prefix, seed, split_script):
    cmd = ['python', split_script, hapmap, pheno, out_prefix, str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"split_between_families.py failed for seed {seed}:\n{result.stderr}")
    tr = re.search(r'Train:\s*(\d+) individuals across (\d+) families', result.stdout)
    te = re.search(r'Test:\s*(\d+) individuals across (\d+) families', result.stdout)
    if not tr or not te:
        raise RuntimeError(f"Could not parse split output for seed {seed}:\n{result.stdout}")
    return {'n_train': int(tr.group(1)), 'n_families_train': int(tr.group(2)),
            'n_test': int(te.group(1)), 'n_families_test': int(te.group(2))}


def run_variants2bin(genotype_file, phenotype_file, phenotype_name, work_subdir, variants2bin_script):
    os.makedirs(work_subdir, exist_ok=True)
    geno_dest = os.path.join(work_subdir, os.path.basename(genotype_file))
    pheno_dest = os.path.join(work_subdir, os.path.basename(phenotype_file))
    shutil.copy(genotype_file, geno_dest)
    shutil.copy(phenotype_file, pheno_dest)
    cmd = ['python', os.path.abspath(variants2bin_script),
           '--genotype_file', os.path.basename(geno_dest),
           '--phenotype_file', os.path.basename(pheno_dest),
           '--phenotype_name', phenotype_name]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_subdir)
    if result.returncode != 0:
        raise RuntimeError(f"variants2bin.py failed in {work_subdir}:\n{result.stderr}")
    bin_path = os.path.join(work_subdir, f'{phenotype_name}.variants2.bin')
    if not os.path.exists(bin_path):
        raise RuntimeError(f"Expected binary output not found at {bin_path}")
    return bin_path


# ---- projection tokens -------------------------------------------------------
def parse_projection(token):
    """'screen_pca:2000' -> ('screen_pca', 2000); 'pls' -> ('pls', None)."""
    if ':' in token:
        name, param = token.split(':', 1)
        return name, int(param)
    return token, None


def build_features(method, param, Xtr_c, train_yn, Xte_c, FL, seed):
    """Return (feature_sets, eff_rank). feature_sets is a list of (Ztr, Zte);
    length 1 for every method except bagging."""
    if method == 'bag':
        n_bags = param if param is not None else 10
        return bagged_feature_sets(Xtr_c, Xte_c, FL, n_bags=n_bags, seed=seed), ''
    if method in ('pls', 'pls_mm'):
        ztr, zte = feature_reduce_ext(Xtr_c, train_yn, FL, method=method, eval_x=Xte_c, seed=seed)
        return [(ztr, zte)], effective_rank(ztr)
    if method == 'screen_pca':
        top_k = param if param is not None else 2000
        ztr, zte = feature_reduce_ext(Xtr_c, train_yn, FL, method='screen_pca',
                                      eval_x=Xte_c, seed=seed, top_k=top_k)
        return [(ztr, zte)], ''
    if method == 'hybrid':
        n_pcs = param if param is not None else 50
        ztr, zte = feature_reduce_ext(Xtr_c, train_yn, FL, method='hybrid',
                                      eval_x=Xte_c, seed=seed, n_pcs=n_pcs)
        return [(ztr, zte)], ''
    # pca and any stock method: delegate through the ext shim (which calls stock)
    ztr, zte = feature_reduce_ext(Xtr_c, train_yn, FL, method=method, eval_x=Xte_c, seed=seed)
    return [(ztr, zte)], ''


def gpfn_and_pcr(feature_sets, train_yn, eval_yn, model):
    """Average GPFN and PCR predictions over feature sets (1 set = no averaging),
    then score. Both read the same feature arrays."""
    gpfn_preds, pcr_preds = [], []
    for ztr, zte in feature_sets:
        if ztr.shape[1] != model.feature_length:
            raise RuntimeError(
                f"projection produced {ztr.shape[1]} features, model needs "
                f"{model.feature_length}. Refusing to feed a mismatched input.")
        gpfn_preds.append(gpfn_predict_from_features(ztr, train_yn, zte, model))
        pcr_preds.append(LinearRegression().fit(ztr, train_yn).predict(zte))
    gpfn_pred = np.mean(gpfn_preds, axis=0)
    pcr_pred = np.mean(pcr_preds, axis=0)
    return (pearsonr(eval_yn, gpfn_pred)[0], _spearman_stat(spearmanr(eval_yn, gpfn_pred)),
            pearsonr(eval_yn, pcr_pred)[0], _spearman_stat(spearmanr(eval_yn, pcr_pred)))


def done_pairs(out_csv):
    if not os.path.exists(out_csv):
        return set()
    done = set()
    with open(out_csv, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['seed']), row['projection']))
    return done


def main():
    ap = argparse.ArgumentParser(description='Repeated between-families projection sweep for GPFN + PCR control.')
    ap.add_argument('--hapmap', required=True)
    ap.add_argument('--pheno', required=True)
    ap.add_argument('--phenotype_name', default='yield')
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--projections', required=True,
                    help="comma list, e.g. pca,pls,pls_mm,screen_pca:2000,hybrid:50,bag:10")
    ap.add_argument('--n_repeats', type=int, default=100)
    ap.add_argument('--start_seed', type=int, default=1)
    ap.add_argument('--out_csv', default='proj_results.csv')
    ap.add_argument('--work_dir', default='proj_runs')
    ap.add_argument('--split_script', default='split_between_families.py')
    ap.add_argument('--variants2bin_script', default='parsing/variants2bin.py')
    ap.add_argument('--keep_intermediate', action='store_true')
    ap.add_argument('--baseline_csv', default=None,
                    help="frozen results_<loc>_<trait>.csv; if given, prints per-seed "
                         "GPFN diff for projection 'pca' as the V0 equality gate.")
    args = ap.parse_args()

    projections = [p.strip() for p in args.projections.split(',') if p.strip()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    model.eval()
    FL = model.feature_length
    print(f"Model loaded on {device}. feature_length={FL}, "
          f"feature_selection(attr)={getattr(model, 'feature_selection', None)}", flush=True)
    print(f"Projections: {projections}", flush=True)

    os.makedirs(args.work_dir, exist_ok=True)
    done = done_pairs(args.out_csv)
    if done:
        print(f"Resuming: {len(done)} (seed,projection) pairs already in {args.out_csv}.", flush=True)

    fieldnames = ['seed', 'projection', 'n_train', 'n_test', 'n_families_train', 'n_families_test',
                  'gpfn_pearson', 'gpfn_spearman', 'pcr_pearson', 'pcr_spearman', 'eff_rank',
                  'wall_clock_seconds']
    write_header = not os.path.exists(args.out_csv)

    seeds = list(range(args.start_seed, args.start_seed + args.n_repeats))
    baseline = {}
    if args.baseline_csv and os.path.exists(args.baseline_csv):
        with open(args.baseline_csv, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    baseline[int(row['seed'])] = float(row['gpfn_pearson'])
                except (KeyError, ValueError):
                    pass

    with open(args.out_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader(); f.flush()

        for si, seed in enumerate(seeds):
            todo = [p for p in projections if (seed, p) not in done]
            if not todo:
                continue

            t0 = time.time()
            print(f"\n[seed {si+1}/{len(seeds)}] seed={seed}  projections={todo}", flush=True)
            seed_dir = os.path.join(args.work_dir, f'seed{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            out_prefix = os.path.join(seed_dir, f'split_seed{seed}')

            try:
                info = run_split(args.hapmap, args.pheno, out_prefix, seed, args.split_script)
                train_bin = run_variants2bin(
                    f'{out_prefix}_btrain.hmp.txt', f'{out_prefix}_btrain_phenotypes.tsv',
                    args.phenotype_name, os.path.join(seed_dir, 'train'), args.variants2bin_script)
                test_bin = run_variants2bin(
                    f'{out_prefix}_btest.hmp.txt', f'{out_prefix}_btest_truevalues.tsv',
                    args.phenotype_name, os.path.join(seed_dir, 'test'), args.variants2bin_script)

                train_x, train_y, _ = load(train_bin)
                eval_x, eval_y, _ = load(test_bin)
                train_x = np.asarray(train_x, dtype=np.float32)
                eval_x = np.asarray(eval_x, dtype=np.float32)
                train_yn = normalize(np.asarray(train_y, dtype=np.float64))
                eval_yn = normalize(np.asarray(eval_y, dtype=np.float64))

                # centre markers ONCE, exactly as stock forward() does
                Xtr_c = center_markers(train_x.copy())
                Xte_c = center_markers(eval_x.copy())

                for proj in todo:
                    method, param = parse_projection(proj)
                    feats, er = build_features(method, param, Xtr_c, train_yn, Xte_c, FL, seed)
                    gp, gs, pp, ps = gpfn_and_pcr(feats, train_yn, eval_yn, model)
                    writer.writerow({
                        'seed': seed, 'projection': proj,
                        'n_train': info['n_train'], 'n_test': info['n_test'],
                        'n_families_train': info['n_families_train'],
                        'n_families_test': info['n_families_test'],
                        'gpfn_pearson': round(gp, 6), 'gpfn_spearman': round(gs, 6),
                        'pcr_pearson': round(pp, 6), 'pcr_spearman': round(ps, 6),
                        'eff_rank': er, 'wall_clock_seconds': round(time.time() - t0, 1)})
                    f.flush()
                    tag = ''
                    if proj == 'pca' and seed in baseline:
                        tag = f"  [frozen {baseline[seed]:.4f}, diff {gp - baseline[seed]:+.4f}]"
                    print(f"    {proj:16s} GPFN r={gp:.4f}  PCR r={pp:.4f}"
                          f"{'  eff_rank='+str(er) if er != '' else ''}{tag}", flush=True)

                if not args.keep_intermediate:
                    shutil.rmtree(seed_dir, ignore_errors=True)

            except Exception as e:
                print(f"  FAILED seed {seed}: {e}", file=sys.stderr, flush=True)
                print("  Continuing. Re-run with the same --out_csv to retry missing pairs.", flush=True)
                continue

    print(f"\nDone. Results in {args.out_csv}", flush=True)


if __name__ == '__main__':
    main()
