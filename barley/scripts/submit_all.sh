#!/bin/bash
# Submit all 9 barley targets as parallel SLURM jobs on PARAM Siddhi.
cd ~/paper1_heb25/scripts
sbatch sbatch_target.sh Dundee_2014_N0 TGW_Dundee_2014_N0_filtered.tsv TGW
sbatch sbatch_target.sh Dundee_2014_N1 TGW_Dundee_2014_N1_filtered.tsv TGW
sbatch sbatch_target.sh Dundee_2015_N0 TGW_Dundee_2015_N0_filtered.tsv TGW
sbatch sbatch_target.sh Dundee_2015_N1 TGW_Dundee_2015_N1_filtered.tsv TGW
sbatch sbatch_target.sh Halle_2014_N0 TGW_Halle_2014_N0_filtered.tsv TGW
sbatch sbatch_target.sh Halle_2014_N1 TGW_Halle_2014_N1_filtered.tsv TGW
sbatch sbatch_target.sh Halle_2015_N0 TGW_Halle_2015_N0_filtered.tsv TGW
sbatch sbatch_target.sh Halle_2015_N1 TGW_Halle_2015_N1_filtered.tsv TGW
sbatch sbatch_target.sh FT FT_BLUEs_filtered.tsv FT
