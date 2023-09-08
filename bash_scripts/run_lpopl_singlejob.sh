#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=10G
#SBATCH -t 99:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=xinyu_liu@brown.edu

export PYTHONUNBUFFERED=TRUE

algo="lpopl"
train_type="mixed"
train_size=50
map=0
prob=0.6
total_steps=800000
incremental_steps=50000
save_dpath="$HOME/data/shared/ltl-transfer"

module load anaconda/2022.05
source /oscar/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate ltl_transfer

python src/run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --map=$map --prob=$prob --total_steps=$total_steps --incremental_steps=$incremental_steps --save_dpath=$save_dpath
