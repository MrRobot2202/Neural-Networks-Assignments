#!/bin/bash
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=2GB
#SBATCH --time=01:00:00

# start environments and load existing modules
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

# ex1_step1
python3 main.py