#!/bin/bash
#SBATCH -n 369
#SBATCH --mem=199G
#SBATCH -t 1:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

export PYTHONUNBUFFERED=TRUE

algo="zero_shot_transfer"
train_type="no_orders"
train_size=50
test_type="mixed"
map=0
run_id=0
relabel_method="cluster"

module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate lpopl
#module load mpi/openmpi_4.0.5_gcc_10.2_slurm20 gcc/10.2 cuda/11.1.1

python run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --test_type=$test_type --map=$map --run_id=$run_id --relabel_method=$relabel_method
#srun --mpi=pmix python -m mpi4py.futures run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --test_type=$test_type --map=$map --run_id=$run_id --relabel_method=$relabel_method
#cp -r ../tmp/* ~/data_gdk/shared/ltl-transfer/tmp/
#cp -r ../results/* ~/data_gdk/shared/ltl-transfer/results/
