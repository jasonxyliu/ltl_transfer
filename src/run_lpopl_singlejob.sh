#!/bin/bash
#SBATCH -n 16
#SBATCH --mem=99G
#SBATCH -t 99:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

export PYTHONUNBUFFERED=TRUE

algo="lpopl"
train_type="hard"
train_size=50
map=1
train_steps=50000

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl

python run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --map=$map --train_steps=$train_steps
