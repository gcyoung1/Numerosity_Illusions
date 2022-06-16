#!/bin/bash
#
#SBATCH --job-name=save_tuning_curves
#SBATCH --output=jobs/save_tuning_curves_%j.txt
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

source activate numerosity_illusions

python -m scripts.analysis.save_tuning_curves --model_directory alexnet_random --layer features_12 --numerosity_neurons_dataset_directory no_illusion --selection_method variance --activations_dataset_directory barbell

python -m scripts.analysis.save_tuning_curves --model_directory alexnet_random --layer features_12 --numerosity_neurons_dataset_directory no_illusion --selection_method variance --activations_dataset_directory no_illusion
