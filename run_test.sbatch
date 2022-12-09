#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=8:00:00
#SBATCH -p owners
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --output=jobs/%j.out

source $GROUP_HOME/software/jwang003/miniconda3/etc/profile.d/conda.sh
conda activate genre-classification

echo "Activated Environment"
python --version
python -u test.py

echo "Finished Job"
