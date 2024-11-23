#!/bin/bash -l

#SBATCH --job-name=xxxx
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:0:0
#SBATCH --export=NONE
#SBATCH --output=messages/%x.out
#SBATCH --error=messages/%x.err

unset SLURM_EXPORT_ENV

module load python/3.10-anaconda
source /home/hpc/iwi5/iwi5253h/venv/bin/activate

cd ~

srun python3 -m src.models.autoencoder --config-path /home/hpc/iwi5/iwi5253h/config/autoencoder.yaml
