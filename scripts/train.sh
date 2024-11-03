#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:4
#SBATCH --job-name=particle-training
#SBATCH --output=job_output_%j.log      
#SBATCH --error=job_error_%j.log        
#SBATCH --ntasks=1                      
#SBATCH --time=00:15:00

module load python/3.10

source ~/.zshrc

cd /home/bpelok/particle-collision
python main.py -e 2

deactivate