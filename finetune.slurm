#!/bin/bash
#SBATCH --job-name=pubmed # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G # Memory allocation
#SBATCH --time=00:30:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=<netid>@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --account=<account>
##SBATCH --partition=<partition>

module purge
module load anaconda3/2024.10
conda activate 202504_llm

mkdir -p /home/$USER/scratch/llmc_test_run_1/logs/wandb
tune run lora_finetune_single_device \
    --config /home/$USER/scratch/GitHub/2025-04-llm-collective/finetune.yaml \
    epochs=1