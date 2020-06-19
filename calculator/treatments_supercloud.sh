#!/bin/bash
#SBATCH -c 40
#SBATCH --array=1-20
#SBATCH --time=1-00:00
#SBATCH --mail-user=agniorf@mit.edu
#SBATCH --mail-type=END
#SBATCH --output=Output.%A_%a.out
source /etc/profile
module load anaconda/2020a
source "/state/partition1/llgrid/pkg/anaconda/anaconda3-2020a/etc/profile.d/conda.sh"
unset PYTHONPATH
export PYTHONNOUSERSITE=True
conda activate covid19
cd covid19_calculator/calculator/
python treatment_cluster.py 'xgboost'