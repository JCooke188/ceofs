#!/bin/bash

# Job Inputs
#SBATCH --job-name=eta_ref
#SBATCH --output=test.out
#SBATCH --ntasks=128
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --partition=cpu,uri-cpu
#SBATCH --mem=900GB

# load necessary python modules
module load python/3.10

source ../../.ceofs/bin/activate

python3 get_eta_ref.py
