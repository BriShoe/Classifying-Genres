#!/bin/bash
#SBATCH --job-name=isomap_nn
#SBATCH --time=8:00:00
#SBATCH -p owners
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --output=jobs/isomap/%j.out

source $GROUP_HOME/software/jwang003/miniconda3/etc/profile.d/conda.sh
conda activate genre-classification

echo "Activated Environment"

cd train || exit
python -u train_multilabel_nn_isomap.py

echo "Finished Job"