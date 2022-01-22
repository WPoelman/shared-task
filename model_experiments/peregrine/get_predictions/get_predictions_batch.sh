#!/bin/bash
#SBATCH --job-name=get_predictions
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
pip install --user -r requirements.txt
python get_predictions.py \
    --input_path $1 \
    --model_path $2 \
    --original_model $3 \
    --output_path $4
