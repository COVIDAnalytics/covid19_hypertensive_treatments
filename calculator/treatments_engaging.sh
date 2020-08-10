#!/bin/bash
#SBATCH --array=1-42
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-18:00:00
#

module load python/3.6.3
module load julia/1.2.0

export IAI_LICENSE_FILE="$HOME/iai.lic"

# python3 treatment_cluster.py 'kn','mlp','qda','gb'
#python3 treatment_cluster.py 'xgboost','rf','cart','lr','oct'
python3 treatment_cluster_controlled.py 'oct'
