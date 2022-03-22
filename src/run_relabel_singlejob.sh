#!/bin/bash
#SBATCH -n 96
#SBATCH --mem=98G
#SBATCH -t 40:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

algo="zero_shot_transfer"
task="transfer_interleaving"
map=0

#export PATH=/users/ashah137/anaconda/lpopl/bin:$PATH

module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
python3
cp -r ../tmp/* ~/data/shared/ltl-transfer/tmp/
