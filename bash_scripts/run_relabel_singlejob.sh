#!/bin/bash
#SBATCH -n 365
#SBATCH --mem=199G
#SBATCH -t 99:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

export PYTHONUNBUFFERED=TRUE

algo="zero_shot_transfer"
train_type="mixed"
train_size=50
test_type="mixed"
map=5
edge_matcher="relaxed"
run_id=0
relabel_method="cluster"
save_dpath="$HOME/data/shared/ltl-transfer"

module load anaconda/2022.05
source /oscar/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate ltl_transfer
module load mpi/openmpi_4.0.7_gcc_10.2_slurm22 gcc/10.2 cuda/11.7.1

#python src/run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --test_type=$test_type --map=$map --run_id=$run_id --relabel_method=$relabel_method --edge_matcher=$edge_matcher --save_dpath=$save_dpath
srun --mpi=pmix python -m mpi4py.futures src/run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --test_type=$test_type --map=$map --run_id=$run_id --relabel_method=$relabel_method --edge_matcher=$edge_matcher --save_dpath=$save_dpath
#cp -r ../tmp/* ~/data_gdk/shared/ltl-transfer/tmp/
#cp -r ../results/* ~/data_gdk/shared/ltl-transfer/results/
#relabel: 365 cores, 200G, 100hrs
#transfer: 100 cores, 100G 90mins
