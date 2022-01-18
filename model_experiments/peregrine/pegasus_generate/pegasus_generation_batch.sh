#!/bin/bash
#SBATCH --job-name=pegasus_generation
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8000

module load Python/3.9.5-GCCcore-10.3.0
pip install --upgrade pip --user
pip install -r requirements.txt --user
python generate_pegasus_data.py \
    --data_directory /home/$USER/shared-task/data_experiments/combined_data \
    --output_dir /data/$USER/shared-task-output/pegasus_generation
