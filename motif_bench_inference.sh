#!/bin/bash -l

# Job Flags
#SBATCH -p sched_mit_sloan_gpu_r8 --gres=gpu:a100:1
#SBATCH --array=1-30
#SBATCH -o slurm_logs/motif_%a.out

conda activate sid_protein_env

python proteinfoundation/motif_inference.py --motif_task_number $SLURM_ARRAY_TASK_ID
