#!/bin/bash
#
#SBATCH --job-name=gen_barbell
#SBATCH --output=../../jobs/gen_barbell_%j.txt
#
#SBATCH --time=3:00:00

ml load math
ml py-scipy/1.4.1_py36
ml py-pytorch/1.4.0_py36

python3 -u gen_barbell_dots.py --dataset-name 5_to_18_2_connection --illusory --num-dots 5 18 --num-pics-per-category 201 --num-train-pics-per-category 1 --connecting-lines 2 --line-length-range 3 400 --line-dist 3 --line-width 1 --conditions 'totalareacontrolsame' 'totalareacontroldifferent' 'dotareacontrol' 'random'
