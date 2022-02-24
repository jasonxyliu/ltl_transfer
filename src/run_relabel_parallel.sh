#!/bin/bash
#SBATCH -n 32
#SBATCH --mem=98G
#SBATCH -t 20:00:00
#SBATCH --array=0

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 1`
j=`expr $SLURM_ARRAY_TASK_ID / 1`
k=`expr $j % 1`
l=`expr $j / 1`
m=`expr $l % 1`

algos=( "zero_shot_transfer" )
algo=${algos[$i]}
tasks=( "transfer_interleaving" )
task=${tasks[$k]}
maps=( 0 )
map=${maps[$m]}

#export PATH=/users/ashah137/anaconda/lpopl/bin:$PATH
module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
python3 run_experiments.py --algorithm=$algo --tasks=$task --map=$map 
cp -r ../tmp/* ~/data/shared/ltl-transfer/tmp/
