#!/bin/bash
#
#SBATCH --job-name=gen_enclosed
#SBATCH --output=../../jobs/gen_enclosed_%j.txt
#
#SBATCH --time=3:00:00

ml load math
ml py-scipy/1.4.1_py36
ml py-pytorch/1.4.0_py36

python3 -u gen_enclosed_dots.py --dataset-name 5_to_18_2_connection --num-dots 5 18 --num-pics-per-category 201 --num-train-pics-per-category 1 --connecting-ellipses 2 --ellipse-ellipse-dist 3 --ellipse-dot-dist 6 --ellipse-width 2 --conditions 'totalareacontrolsame' 'totalareacontroldifferent' 'dotareacontrol' 'random'

#python gen_enclosed_dots.py --dataset-name sdev --num-dots 5 18 --num-pics-per-category 1 --num-train-pics-per-category 1 --connecting-ellipses 2 --ellipse-ellipse-dist 3 --ellipse-dot-dist 3 --ellipse-width 2 --conditions 'totalareacontrolsame'
