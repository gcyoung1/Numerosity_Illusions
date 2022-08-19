#!/bin/bash
#
#SBATCH --job-name=identify_numerosity_neurons
#SBATCH --output=jobs/identify_numerosity_neurons_%j.txt
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem 192G

source activate numerosity_illusions

#python -m scripts.analysis.identify_numerosity_neurons --model_directory alexnet_random --dataset_directory no_illusion_dewind_circles_05-27-2022:22_48 --layer features_12 --selection_method variance

python -m scripts.analysis.identify_numerosity_neurons --model_directory alexnet_pretrained --dataset_directory no_illusion_dewind_circles_05-27-2022:22_48 --layer features_12 --selection_method variance
