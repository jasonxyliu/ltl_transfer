#!/bin/bash
#SBATCH -n 99
#SBATCH --mem=199G
#SBATCH -t 36:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

export PYTHONUNBUFFERED=TRUE

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl

python visualize_dfa_graph.py
