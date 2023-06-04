#!/bin/bash -l
#SBATCH --job-name="lstm"
#SBATCH --time 16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=32G

module load cuda/11.7.0
module load python/3.7.7
source ../../venv_nlp/bin/activate

python train_lstm.py  