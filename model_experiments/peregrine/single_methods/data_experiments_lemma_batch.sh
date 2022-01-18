#!/bin/bash
#SBATCH --job-name=data_experiment_job
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=4000

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
pip install --user -r requirements.txt
python data_experiments.py \
    --data_directory /home/$USER/shared-task/data_experiments/lemma_test  \
    --test_set /home/$USER/shared-task/en_new_test_split_lemma.csv \
    --output_dir /data/$USER/shared-task-output
