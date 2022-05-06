#!/bin/bash
#
#SBATCH --job-name=gen_stimuli
#SBATCH --output=../../jobs/gen_stimuli_%j.txt
#
#SBATCH --time=3:00:00

source activate numerosity_illusions

python -u gen_dewind_circles.py --pic_width 227 --pic_height 227 --dataset-name no_illusion --num-pics-per-category 10 --numerosities 2 3 4 5 6 --sizes 13 14 15 16 17 --spacings 20 21 22 23 24 

python -u gen_dewind_circles.py --pic_width 227 --pic_height 227 --dataset-name hollow --num-pics-per-category 10 --numerosities 2 3 4 5 6 --sizes 13 14 15 16 17 --spacings 20 21 22 23 24 --hollow

python -u gen_dewind_circles.py --pic_width 227 --pic_height 227 --dataset-name barbell --num-pics-per-category 10 --numerosities 2 3 4 5 6 --sizes 13 14 15 16 17 --spacings 20 21 22 23 24 --num_lines 0 1 2 --line_length_range 3 400

python -u gen_dewind_circles.py --pic_width 227 --pic_height 227 --dataset-name illusory_contour --num-pics-per-category 10 --numerosities 2 3 4 5 6 --sizes 13 14 15 16 17 --spacings 20 21 22 23 24 --num_lines 0 1 2 --line_length_range 3 400 --illusory

