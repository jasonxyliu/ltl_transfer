#!/bin/bash
#SBATCH -n 96
#SBATCH --mem=256G
#SBATCH -t 10:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

export PYTHONUNBUFFERED=TRUE

algo="zero_shot_transfer"
task="transfer_interleaving"
map=0

#export PATH=/users/ashah137/anaconda/lpopl/bin:$PATH

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
module load mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2 cuda/11.1.1

srun --mpi=pmix python -m mpi4py.futures run_experiments.py --algorithm=$algo --tasks=$task --map=$map
cp -r ../tmp/* ~/data/shared/ltl-transfer/tmp/