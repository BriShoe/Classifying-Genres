#!/bin/bash
#SBATCH --job-name=baseline_nn
#SBATCH --time=8:00:00
#SBATCH -p owners
#SBATCH -c 2
#SBATCH --mem=16GB
#SBATCH --output=jobs/baseline/%j.out

source $GROUP_HOME/software/jwang003/miniconda3/etc/profile.d/conda.sh
conda activate genre-classification

echo "Activated Environment"

python train/train_multilabel_nn.py

echo "Finished Job"

