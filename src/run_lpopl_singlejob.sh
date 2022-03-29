#!/bin/bash
#SBATCH -n 64
#SBATCH --mem=198G
#SBATCH -t 96:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

export PYTHONUNBUFFERED=TRUE

algo="lpopl"
task="no_orders"
train_size=50
test_task="no_orders"
map=0

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
module load mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2 cuda/11.1.1

srun --mpi=pmix python -m mpi4py.futures run_experiments.py --algo=$algo --tasks=$task --train_size=$train_size --test_tasks=$test_task --map=$map
