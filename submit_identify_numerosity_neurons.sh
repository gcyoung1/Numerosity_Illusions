#!/bin/bash
#
#SBATCH --job-name=gen_stimuli
#SBATCH --output=jobs/gen_stimuli_%j.txt
#
#SBATCH --time=3:00:00

source activate numerosity_illusions

python -m scripts.analysis.identify_numerosity_neurons --model_directory alexnet --dataset_directory no_illusion --layers classifier_6 --selection_method variance

