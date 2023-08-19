#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=5G
#SBATCH -t 24:00:00
#SBATCH --array=0-359

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 360`
j=`expr $SLURM_ARRAY_TASK_ID / 360`

algo="zero_shot_transfer"
task_id=4
map_id=0
run_id=0
#ltl_ids=(seq 34)
#ltl_id=${ltl_ids[$i]}
ltl_id=1
state_ids=(seq 0 360)
state_id=${state_ids[$i]}
n_rollouts=100
max_depth=100
save_dpath="$HOME/data/shared/ltl-transfer"

module load anaconda/2022.05
source /oscar/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lpopl

python3 run_single_worker.py --algo=$algo --task_id=$task_id --map_id=$map_id --run_id=$run_id --ltl_id=%ltl_id --state_id=%state_id --n_rollouts=%n_rollouts --max_depth=%max_depth --save_dpath=$save_dpath
cp -r ../tmp/* ~/data/shared/ltl-transfer/tmp/
