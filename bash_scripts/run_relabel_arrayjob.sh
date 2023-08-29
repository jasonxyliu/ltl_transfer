#!/bin/bash
#SBATCH -n 119
#SBATCH --mem=99G
#SBATCH -t 199:00:00
#SBATCH --array=0-15

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

export PYTHONUNBUFFERED=TRUE

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 1`
j=`expr $SLURM_ARRAY_TASK_ID / 1`
k=`expr $j % 1`
l=`expr $j / 1`
m=`expr $l % 4`
n=`expr $l / 4`
o=`expr $n % 4`
p=`expr $n / 4`
q=`expr $p % 1`

algo="zero_shot_transfer"

train_type="mixed"

test_types=( "mixed" )  # "hard" "mixed" "soft_strict" "soft" "no_orders"
test_type=${test_types[$i]}

train_sizes=( 50 )  # 5, 10, 15, 20, 30, 40, 50
train_size=${train_sizes[$k]}

maps=( 0 1 5 6 )  # 0 1 5 6
map=${maps[$m]}

probs=( 0.9 0.7 0.6 0.5 )  # 1.0 0.9 0.8 0.7 0.6 0.5
prob=${probs[$o]}

edge_matchers=( "relaxed" )  # "relaxed" "rigid"
edge_matcher=${edge_matchers[$q]}

run_id=0
relabel_method="cluster"
save_dpath="$HOME/data/shared/ltl-transfer"

module load anaconda/2022.05
source /oscar/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate ltl_transfer
module load mpi/openmpi_4.1.1_gcc_10.2_slurm22

srun --mpi=pmix python -m mpi4py.futures src/run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --test_type=$test_type --map=$map --run_id=$run_id --relabel_method=$relabel_method --edge_matcher=$edge_matcher --save_dpath=$save_dpath
