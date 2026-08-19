# Sparse marker effects expose a projection bottleneck in Prior-Data Fitted Networks for genomic prediction

Code, per-partition results, and figure scripts for the manuscript

> G. Tiwari, A. Tiwari. *Sparse Marker Effects Expose a Projection Bottleneck in Prior-Data Fitted Networks for Genomic Prediction, Confirmed Across Two Species.* Submitted to IEEE Transactions on Computational Biology and Bioinformatics.

## What this is

We re-evaluated Genomic Prior-Data Fitted Networks (GPFN, Ubbens et al. 2025, doi:10.1109/TCBBIO.2025.3596744) on a between-families genomic prediction task, adding BayesB to a baseline set that had included GBLUP, principal components regression and XGBoost.

Two panels, four methods, 100 independent partitions each, every method evaluated on byte-identical splits.

| Panel | Design | Fits |
|---|---|---|
| SoyNAM soybean | 17 location-trait combinations x 100 partitions x 4 methods | 6,800 |
| Barley HEB-25 | 9 targets x 100 partitions x 4 methods | 3,600 |
| | **Total** | **10,400** |

**Headline result.** BayesB is more accurate than GPFN in 16 of 17 soybean combinations, significant in 15 after Holm-Bonferroni correction. GPFN ranks last of four and does not exceed principal components regression, the ablation introduced by GPFN's own authors. The cost is architecture-conditional: in barley it is zero across eight polygenic grain-weight targets and large on the single oligogenic trait.

### Replacing the projection at inference does not recover the loss

If the fixed principal-component projection is the limitation, the obvious remedy is to swap it, at inference and without refitting the prior, for one that keeps sparse large-effect signal. On IL protein, the combination with one of the largest deficits, we replaced the released projection with four trait-aware alternatives (PLS, PLS matched to the principal-component variance profile, association-screen-then-PCA, a hybrid, and feature bagging), each fitted on the training partition only, across the same 100 partitions. None beats the released projection: screen-then-PCA comes closest at a paired deficit of 0.012 (significant after Holm-Bonferroni), the rest by more. The screened features carry the signal (a linear model reads them as well as the principal components, and GPFN on them tracks that linear control at r = 0.89), yet GPFN does not exceed what it achieves on the basis it was fitted against. See Table III in `analysis/reproduce_all_tables.py` and `soybean/figures/Fig12_projection_swap.pdf`.

## Reproduce every number in the paper

```bash
git clone https://github.com/gaurav-iitindore/gpfn-projection-bottleneck.git
cd gpfn-projection-bottleneck
pip install -r requirements.txt
python analysis/reproduce_all_tables.py
```

This reads only the raw per-partition CSVs and prints Tables I to VII plus every inline statistic, in manuscript order. No intermediate summary files are used. Runtime is a few seconds.

## Layout

```
soybean/
  scripts/                 analysis and figure generation
    proj_ext.py                    trait-aware projection variants
                                   (PLS, screen-then-PCA, hybrid, bagging)
    run_projection_evaluation.py   runs GPFN + a PCR control on byte-identical
                                   features, one projection at a time, resumable
    make_fig12_projection.py       regenerates Figure 12 from the raw CSV
  results/
    final_results/         17 CSVs, one per location-trait combination,
                           100 rows each, one row per partition,
                           columns: seed, n_train, n_test, and
                           {gpfn,gblup,pcr,bayesb}_{pearson,spearman}
    heritability.csv       genomic h2 per combination (REML on a VanRaden GRM)
    projection/
      proj_IL_2012_protein_100.csv IL protein, 100 partitions, one row per
                                   (seed, projection): gpfn/pcr pearson+spearman
  figures/                 Figures 1 to 7 and Figure 12 as vector PDF

barley/
  scripts/                 analysis, figures, SLURM submission
  results/
    raw_results/           py_<target>.csv  GPFN, GBLUP, PCR (Python)
                           r_<target>.csv   BayesB (R, BGLR)
                           9 targets, 100 rows each
    compute_time.txt       wall-clock and CPU-hour accounting
  figures/                 Figures 8 to 11 as vector PDF
  DATA_PROVENANCE.md       genotype and phenotype sources, QC decisions

analysis/
  reproduce_all_tables.py  regenerates every reported number from raw data
```

## What is not here

**Genotype and phenotype matrices.** Both are redistributable from their original sources and are too large for this repository.

- **SoyNAM.** Imputed genotypes `soynam_29416_imputed.hmp.txt` (Wm82.gnm2.div.Song_NAM_2021a) from SoyBase, produced by Chen et al. 2022 from SoySNP50K assays for 5,176 lines. Phenotypes from the SoyNAM project, 2012 season. After QC, 29,131 markers.
- **Barley HEB-25.** Imputed marker matrix (33,005 markers) and BLUEs from Maurer et al. 2015, doi:10.1186/s12864-015-1459-7. Physical positions via the barley 50k iSelect array, Bayer et al. 2017, doi:10.3389/fpls.2017.01792, against the Morex reference. 32,833 markers retained after identifier reconciliation. See `barley/DATA_PROVENANCE.md`.

**GPFN model weights and implementation.** Released by Ubbens and colleagues at https://github.com/jubbens/gpfn and used here **unmodified**. We used `pika.pt`, the model designated for structured populations, for both panels. GPFN, GBLUP and PCR were all run with the authors' own evaluation code. Nothing we report about those three methods can be attributed to a re-implementation on our part.

## Method configuration

| Method | Implementation | Notes |
|---|---|---|
| GPFN | Ubbens et al., unmodified | `pika.pt`, 311M parameters, NAM prior. Input is the first 100 PCs, fitted on the training partition only |
| GBLUP | Ubbens et al., unmodified | VanRaden GRM over all markers, heritability fixed at 0.5 |
| PCR | Ubbens et al., unmodified | OLS on the same 100 PCs supplied to GPFN |
| BayesB | BGLR (Pérez and de los Campos 2014) | 12,000 Gibbs iterations, 2,000 burn-in. Test phenotypes set to `NA` during sampling |

Partitions are a deterministic function of the seed, so every method within a replicate saw exactly the same training and test individuals. Partition metadata was compared seed by seed across all combinations before any results were merged.

## Known data caveat

`summary_4method.csv` files, where present, carry an `n_train` column recording a **single partition** rather than the mean across the 100 partitions. Those values are 14 to 19 individuals too high for barley. The manuscript and `analysis/reproduce_all_tables.py` both compute the mean from the raw files. Do not cite the summary files for training sizes.

## Reproducibility

The deterministic methods (GPFN, GBLUP, PCR) reproduce to four decimal places across two independent clusters with independently built, version-pinned Conda environments. BayesB reproduces within its expected sampling distribution, as a Gibbs sampler must.

Soybean was run on a single node: 30 CPU cores, 488 GB RAM, one NVIDIA H200. Barley was run on PARAM Siddhi-AI (C-DAC).

## Citing

See `CITATION.cff`. If you use the GPFN model itself, cite Ubbens, Stavness and Sharpe (2025) as well.

## License

Code and results in this repository are released under the MIT License. See `LICENSE`. Third-party data referenced above remain under their original licenses.
