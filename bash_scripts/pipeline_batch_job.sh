#!/bin/bash -l

#SBATCH --job-name=PIPE_1
#SBATCH --clusters=woody
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=24:0:0
#SBATCH --export=NONE
#SBATCH --output=messages/%x.out
#SBATCH --error=messages/%x.err

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
source /home/hpc/iwfa/iwfa106h/package/woody_venv/bin/activate

NAME="PIPE_1"

cd /home/hpc/iwfa/iwfa106h/package/

srun python3 -m src.pipelines.dataset_model_pipeline --config-path /home/hpc/iwfa/iwfa106h/config/experiments/"${NAME}.yaml"
