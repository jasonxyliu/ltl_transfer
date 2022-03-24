#!/bin/bash
#SBATCH -n 2
#SBATCH --mem=298G
#SBATCH -t 96:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

algo="lpopl"
task="no_orders"
train_size=10
test_task="no_orders"
map=0

module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
python3 run_experiments.py --algo=$algo --tasks=$task --train_size=$train_size --test_tasks=$test_task --map=$map
