#!/bin/bash -l

# Job Flags
#SBATCH -p sched_mit_sloan_gpu_r8 --gres=gpu:a100:1 -t 24:00:00 --mem=256G
#SBATCH -o pdb_run.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

conda init
conda activate sid_protein_env

rm -rf store/train_run_pdb_finetune_single/
python proteinfoundation/train.py --config_name finetuning_ca --show_prog_bar --single
