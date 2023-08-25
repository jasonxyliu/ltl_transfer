#!/bin/bash
#SBATCH -n 119
#SBATCH --mem=99G
#SBATCH -t 5:00:00
#SBATCH --array=0-4

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

export PYTHONUNBUFFERED=TRUE

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 5`
j=`expr $SLURM_ARRAY_TASK_ID / 5`
k=`expr $j % 1`
l=`expr $j / 1`
m=`expr $l % 1`
n=`expr $l / 1`
o=`expr $n % 1`

algo="zero_shot_transfer"

train_type="mixed"

test_types=( "hard" "mixed" "soft_strict" "soft" "no_orders" )
test_type=${test_types[$i]}

train_sizes=( 50 )  # 5, 10, 15, 20, 30, 40, 50
train_size=${train_sizes[$k]}


maps=( 1 )  # 1 5 6
map=${maps[$m]}

edge_matchers=( "relaxed" )  # "relaxed" "rigid"
edge_matcher=${edge_matchers[$o]}

run_id=0
relabel_method="cluster"
save_dpath="$HOME/data/shared/ltl-transfer"

module load anaconda/2022.05
source /oscar/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate ltl_transfer
module load mpi/openmpi_4.0.7_gcc_10.2_slurm22 gcc/10.2 cuda/11.7.1

srun --mpi=pmix python -m mpi4py.futures src/run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --test_type=$test_type --map=$map --run_id=$run_id --relabel_method=$relabel_method --edge_matcher=$edge_matcher --save_dpath=$save_dpath
# cp -r ../tmp/* ~/data_gdk/shared/ltl-transfer/tmp/
# cp -r ../results/* ~/data_gdk/shared/ltl-transfer/results/
