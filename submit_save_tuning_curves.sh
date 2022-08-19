#!/bin/bash
#
#SBATCH --job-name=save_tuning_curves
#SBATCH --output=jobs/save_tuning_curves_%j.txt
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem 192G

source activate numerosity_illusions

for model in alexnet_random alexnet_pretrained;
do echo $model;
for dataset in `ls data/stimuli`; 
do echo $dataset; 
python -m scripts.analysis.save_tuning_curves --model_directory $model --layer features_12 --numerosity_neurons_dataset_directory no_illusion_dewind_circles_05-27-2022:22_48 --selection_method variance --activations_dataset_directory $dataset;
done;
done
