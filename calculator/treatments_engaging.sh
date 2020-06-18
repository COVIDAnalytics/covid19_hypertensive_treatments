#!/bin/bash
#SBATCH --array=1-2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=1-00:00:00
#

module load python/3.6.3
module load julia/1.2.0

export IAI_LICENSE_FILE="$HOME/iai.lic"

python3 src/treatment_cluster_oct.py