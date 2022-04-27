#!/bin/bash
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -t 99:00:00
#SBATCH --array=0-14

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

export PYTHONUNBUFFERED=TRUE

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 5`
j=`expr $SLURM_ARRAY_TASK_ID / 5`
k=`expr $j % 3`
#l=`expr $j / 3`
#m=`expr $l % 1`

algo="lpopl"

train_types=( "hard" "mixed" "soft_strict" "soft" "no_orders" )
train_type=${train_types[$i]}

train_size=50

maps=(1 5 6)
map=${maps[$k]}

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl

python run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --map=$map
