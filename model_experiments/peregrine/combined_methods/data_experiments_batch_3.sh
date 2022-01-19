#!/bin/bash
#SBATCH --job-name=dataset_3
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8000

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
pip install --user -r requirements.txt
python data_experiments.py \
    --data_directory /home/$USER/shared-task/data_experiments/combined_methods/dataset_3 \
    --test_set /home/$USER/shared-task/data_experiments/en_new_test_split.csv \
    --output_dir /data/$USER/shared-task-output/combined_methods
