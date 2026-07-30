#!/bin/bash
# PARAM Siddhi SLURM job: run one target (Python methods + BayesB), 100 seeds.
# Usage: sbatch sbatch_target.sh <NAME> <PHENO_FILE> <TGW|FT>
#SBATCH --job-name=heb
#SBATCH --partition=cpup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.log

set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gpfn
cd ~/gpfn
export PYTHONPATH=~/gpfn:${PYTHONPATH:-}

NAME=$1; PHENO=$2; PN=$3
HAP=~/paper1_heb25/geno/HEB25_mapped.hmp.txt
RES=~/paper1_heb25/results
RSCRIPT=~/miniforge3/envs/rgs/bin/Rscript

# Python methods: GPFN, GBLUP, PCR
python run_repeated_evaluation.py \
  --hapmap $HAP --pheno ~/paper1_heb25/pheno/$PHENO --model_path deploy/pika.pt \
  --n_repeats 100 --start_seed 1 --phenotype_name $PN \
  --out_csv $RES/py_${NAME}.csv --work_dir $RES/work_py_${NAME}

# BayesB (BGLR) via the R runner; GA-GBLUP skipped
python run_r_methods_evaluation.py \
  --hapmap $HAP --pheno ~/paper1_heb25/pheno/$PHENO \
  --n_repeats 100 --start_seed 1 --phenotype_name $PN --methods bayesb \
  --rscript_bin $RSCRIPT \
  --out_csv $RES/r_${NAME}.csv --work_dir $RES/work_r_${NAME}
