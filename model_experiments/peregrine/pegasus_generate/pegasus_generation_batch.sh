#!/bin/bash
#SBATCH --job-name=pegasus_generation
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8000

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
pip install --upgrade pip --user
pip install -r requirements.txt --user
python generate_pegasus_data.py \
    --data_directory /home/$USER/shared-task/combined_data \
    --output_dir /data/$USER/shared-task-output/pegasus_generation
