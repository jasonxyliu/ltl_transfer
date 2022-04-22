#!/bin/bash
#SBATCH -n 100
#SBATCH --mem=99G
#SBATCH -t 2:00:00
#SBATCH --array=0-4

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

export PYTHONUNBUFFERED=TRUE

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 5`
j=`expr $SLURM_ARRAY_TASK_ID / 5`
k=`expr $j % 1`
#l=`expr $j / 1`
#m=`expr $l % 1`

algo="zero_shot_transfer"

train_type="hard"

test_types=( "hard" "mixed" "soft_strict" "soft" "no_orders" )
test_type=${test_types[$i]}

train_sizes=( 50 )
train_size=${train_sizes[$k]}

map=0

run_id=0
relabel_method="cluster"
edge_matcher="relaxed"

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
module load mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2 cuda/11.1.1

srun --mpi=pmix python -m mpi4py.futures run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --test_type=$test_type --map=$map --run_id=$run_id --relabel_method=$relabel_method --edge_matcher=$edge_matcher
cp -r ../tmp/* ~/data_gdk/shared/ltl-transfer/tmp/
cp -r ../results/* ~/data_gdk/shared/ltl-transfer/results/
