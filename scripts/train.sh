#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus-per-node=8
#SBATCH --job-name=particle-training
#SBATCH --output=job_output_%j.log      
#SBATCH --error=job_error_%j.log        
#SBATCH --ntasks=1               
#SBATCH --time=5-00:00:00

module load python/3.10

source ~/.zshrc

cd /home/bpelok/particle-collision
python main.py -d train_big -b 4 --lr 1e-3 --min_lr 1e-7 --ddpm_num_steps 400 --ddpm_processor_layers 1 --pooling_levels 5 --ddpm_use_reverse_posterior

deactivate
