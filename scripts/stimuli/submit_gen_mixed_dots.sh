#!/bin/bash
#
#SBATCH --job-name=gen_mixed
#SBATCH --output=../../jobs/gen_mixed_%j.txt
#
#SBATCH --time=3:20:00

ml load math
ml load labs poldrack
ml load anaconda/5.0.0-py36
source activate numerosity
ml py-pytorch/1.4.0_py36

python -u gen_mixed_dots.py --dataset-name 18 --num-dots 1 18 --num-pics-per-category 201 --num-train-pics-per-category 1 --conditions 'totalareacontrolsame' 'totalareacontroldifferent' 'dotareacontrol' 'random' 
