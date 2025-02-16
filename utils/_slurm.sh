#!/bin/bash

#SBATCH -J bwa
#SBATCH -o slurm.out
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=00:08:00
#SBATCH --exclusive

nvidia-smi
bash scripts/test.sh 100bp 10000
#bash scripts/smalltest.sh 4m.a6000
