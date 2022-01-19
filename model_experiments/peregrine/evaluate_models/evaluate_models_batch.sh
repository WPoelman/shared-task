#!/bin/bash
#SBATCH --job-name=evaluate_models
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
pip install --user -r requirements.txt
python evaluate_models.py \
    --input_path /home/$USER/shared-task/data_experiments/en_new_test_split.csv \
    --starting_path /data/$USER/shared-task-output \
    --output_path /home/$USER/shared-task/evaluation_results_normal_test_set.csv
