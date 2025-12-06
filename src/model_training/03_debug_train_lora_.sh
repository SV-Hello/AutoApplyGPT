#!/bin/bash
#SBATCH --job-name=ModelTraining689       #Set the job name to "JobExample1"
#SBATCH --time=02:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out       
#SBATCH --error=logs/train_%j.err       
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:h100:1

module load GCCcore/14.2.0 Python/3.13.1

cd $SCRATCH
cd 689csce

source .venv/bin/activate

cd src/model_training/
python 03_train_lora.py
