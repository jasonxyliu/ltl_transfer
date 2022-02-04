#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=3G
#SBATCH -t 24:00:00
#SBATCH --array=0-119

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 1`
j=`expr $SLURM_ARRAY_TASK_ID / 1`
k=`expr $j % 2`
l=`expr $j / 2`
m=`expr $l % 1`

algos=( "lpopl" )
algo=${algos[$i]}
tasks=( "transfer_sequence" "transfer_interleaving" )
task=${tasks[$k]}
maps=( 0 )
map=${maps[$m]}

# export PATH=/users/xliu53/anaconda/lpopl/bin:$PATH
module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
python3 run_experiments.py --algorithm=$algo --tasks=$task --map=$map
