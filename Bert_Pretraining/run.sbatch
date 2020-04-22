#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=3:00:00
#SBATCH --job-name="BertMLM"
#SBATCH --mail-user=rgg296@nyu.edu

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

NETID=rgg296
source activate /scratch/${NETID}/nlu/env

python /scratch/rgg296/final_project/BertMLM.py