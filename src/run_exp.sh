#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=3G
#SBATCH -t 24:00:00
#SBATCH --array=0-29

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 3`
j=`expr $SLURM_ARRAY_TASK_ID / 3`
k=`expr $j % 10`
l=`expr $j / 10`

algo="lpopl"
tasks=( "sequence" "interleaving" "safety" )
task=${tasks[$i]}
maps=( 0 1 2 3 4 5 6 7 8 9)
map=${maps[$k]}

# export PATH=/users/xliu53/anaconda/lpopl/bin:$PATH
module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
python3 run_experiments.py --algorithm=$algo --tasks=$task $j --map=$map
