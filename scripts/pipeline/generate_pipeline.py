import yaml
import argparse
import shutil
import os
import time
from ..models import utility_functions as model_utils

def stringify_arg(arg_name, arg_value):
    if type(arg_value) is list:
        arg_value = " ".join([str(x) for x in arg_value])
    return f"--{arg_name} {arg_value}"

def create_gen_stimuli_command(dataset_object):
    base = "python -m scripts.stimuli.gen_dewind_circles"
    if dataset_object['interpolate']:
        for param in ['sizes', 'spacings', 'numerosities']:
            assert len(dataset_object[param]) == 2
            start,end = dataset_object[param]
            assert end >= start
            step_size = (end-start)/dataset_object['num_steps']
            dataset_object[param] = [str(start + i*step_size) for i in range(dataset_object['num_steps']+1)]
    binary_args = ["hollow", "illusory", "linear_args"]
    interpolate_args = ["interpolate", "num_steps"]
    command = base + "".join([f" {stringify_arg(arg_name, arg_value)}" for arg_name, arg_value in dataset_object.items() if arg_name not in interpolate_args+binary_args])
    for binary_arg in binary_args:
        if dataset_object.get(binary_arg, False):
            command += f" --{binary_arg}"
    command += f" --experiment_name {config['experiment_name']}"
    return command
    

def create_gen_stimuli_sbatch(config):
    hours = int(3*((1 + config["numerosity_neurons_dataset"]["num_pics_per_category"])//10))
    header = f"""#!/bin/bash
#
#SBATCH --job-name=1_gen_stimuli_{config['experiment_name']}
#SBATCH --output=jobs/1_gen_stimuli_{config['experiment_name']}_%j.out
#
#SBATCH --time={hours}:00:00

source activate numerosity_illusions;

"""
    numerosity_neurons_command = create_gen_stimuli_command(config['numerosity_neurons_dataset'])
    sbatch = header + numerosity_neurons_command + "\n\n"
    for activations_dataset in config['activations_datasets']:
        activations_dataset_command = create_gen_stimuli_command(activations_dataset)
        sbatch += activations_dataset_command + "\n\n"
    return sbatch

def create_save_layers_sbatch(config):
    dataset_names = [dataset['dataset_name'] for dataset in config['activations_datasets']] + [config['numerosity_neurons_dataset']['dataset_name']]
    time = int(len(config['layers']) * len(dataset_names) * 0.5 * 2) + 1 # Empirically determined
    sbatch = f"""#!/bin/bash
#
#SBATCH --job-name=2_save_layers_{config['experiment_name']}
#SBATCH --output=jobs/2_save_layers_{config['experiment_name']}_%j.txt
#
#SBATCH --time={time}:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH -p gpu
#SBATCH -G 2

source activate numerosity_illusions;

for dataset in {" ".join(dataset_names)};
do echo $dataset; 
python -m scripts.models.save_layers --model {config["model_name"]} --dataset_name $dataset --layers {" ".join(config['layers'])}  --num_workers 1 --experiment_name {config['experiment_name']};
python -m scripts.models.save_layers --model {config["model_name"]} --dataset_name $dataset --layers {" ".join(config['layers'])}  --num_workers 1 --experiment_name {config['experiment_name']} --pretrained;
done
"""
    return sbatch

def create_identify_numerosity_neurons_sbatch(config):
    num_images = config["numerosity_neurons_dataset"]["num_pics_per_category"]*len(config["numerosity_neurons_dataset"]["sizes"])*len(config["numerosity_neurons_dataset"]["spacings"])*len(config["numerosity_neurons_dataset"]["numerosities"])
    num_levels = max(len(config["numerosity_neurons_dataset"]["sizes"]), len(config["numerosity_neurons_dataset"]["spacings"]), len(config["numerosity_neurons_dataset"]["numerosities"]))
    seconds_per_neuron = (0.55*(0.08/1080)*num_images + 0.4)*2.5**(max(num_levels-6,0)) # Empirically determined
    seconds = 0
    dict_layer_size = model_utils.get_layer_size_dict(config["model_name"])
    layer_sizes = [dict_layer_size[layer_name] for layer_name in config['layers']]
    for layer_size in layer_sizes:
        seconds += seconds_per_neuron * layer_size
    hours = int(seconds // 3600)
    hourbuffer = int(hourbuffer * 1.2)
    hours += hourbuffer

    sbatch = f"""#!/bin/bash
#
#SBATCH --job-name=3_identify_numerosity_neurons_{config['experiment_name']}
#SBATCH --output=jobs/3_identify_numerosity_neurons_{config['experiment_name']}_%j.txt
#
#SBATCH --time={hours}:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem 192G

source activate numerosity_illusions;

for model_type in {" ".join(config['model_types'])};
do model={config["model_name"]}_$model_type;
for layer in {" ".join(config['layers'])};
do echo $layer;
python -m scripts.analysis.identify_numerosity_neurons --model_directory $model --dataset_name {config['numerosity_neurons_dataset']['dataset_name']} --layer $layer --selection_method {config["selection_method"]} --experiment_name {config['experiment_name']};
done; 
done
"""
    return sbatch

def create_save_tuning_sbatch(config):
    activations_dataset_names = [dataset['dataset_name'] for dataset in config['activations_datasets']] + [config['numerosity_neurons_dataset']['dataset_name']]
    sbatch = f"""#!/bin/bash
#
#SBATCH --job-name=4_save_tuning_curves_{config['experiment_name']}
#SBATCH --output=jobs/4_save_tuning_curves_{config['experiment_name']}_%j.txt
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

source activate numerosity_illusions

for model_type in {" ".join(config['model_types'])};
do model={config["model_name"]}_$model_type;
echo $model;
for dataset in {" ".join(activations_dataset_names)};
do echo $dataset; 
for layer in {" ".join(config['layers'])};
do echo $layer;
python -m scripts.analysis.save_tuning_curves --model_directory $model --layer $layer --numerosity_neurons_dataset_name {config['numerosity_neurons_dataset']['dataset_name']} --selection_method {config['selection_method']} --activations_dataset_name $dataset --experiment_name {config['experiment_name']};
done;
done;
done
"""
    return sbatch

def create_plot_tuning_sbatch(config):
    activations_dataset_names = [dataset['dataset_name'] for dataset in config['activations_datasets']] + [config['numerosity_neurons_dataset']['dataset_name']]
    time = int(len(activations_dataset_names) * 0.5 * 2) + 1
    sbatch = f"""#!/bin/bash
#
#SBATCH --job-name=5_plot_tuning_curves_{config['experiment_name']}
#SBATCH --output=jobs/5_plot_tuning_curves_{config['experiment_name']}_%j.txt
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

source activate numerosity_illusions

for model_type in {" ".join(config['model_types'])};
do model={config["model_name"]}_$model_type;
echo $model;
for dataset in {" ".join(activations_dataset_names)};
do echo $dataset; 
for layer in {" ".join(config['layers'])};
do echo $layer;
python -m scripts.plotting.plot_tuning_curves --model_directory $model --layer $layer --numerosity_neurons_dataset_name {config['numerosity_neurons_dataset']['dataset_name']} --selection_method {config['selection_method']} --activations_dataset_name $dataset --experiment_name {config['experiment_name']};
done;
done;
done
"""
    return sbatch

def generate_pipeline(config):
    # Create experiment pipeline folder
    os.mkdir(os.path.join('experiment_runs', config['experiment_name']))
    pipeline_folder = os.path.join('experiment_runs', config['experiment_name'])
    # Create sbatch files for each stage of experiment
    gen_stimuli_sbatch = create_gen_stimuli_sbatch(config)
    with open(os.path.join(pipeline_folder, "1_submit_gen_stimuli.sbatch"),"w") as f:
        f.write(gen_stimuli_sbatch)
    save_layers_sbatch = create_save_layers_sbatch(config)
    with open(os.path.join(pipeline_folder, "2_submit_save_layers.sbatch"),"w") as f:
        f.write(save_layers_sbatch)
    identify_numerosity_neurons_sbatch = create_identify_numerosity_neurons_sbatch(config)
    with open(os.path.join(pipeline_folder, "3_identify_numerosity_neurons.sbatch"),"w") as f:
        f.write(identify_numerosity_neurons_sbatch)
    save_tuning_sbatch = create_save_tuning_sbatch(config)
    with open(os.path.join(pipeline_folder, "4_submit_save_tuning.sbatch"),"w") as f:
        f.write(save_tuning_sbatch)
    plot_tuning_sbatch = create_plot_tuning_sbatch(config)
    with open(os.path.join(pipeline_folder, "5_submit_plot_tuning.sbatch"),"w") as f:
        f.write(plot_tuning_sbatch)

if __name__ == '__main__':
    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Generate pipeline for running experiment.')
    parser.add_argument('config_filename', type=str, 
                        help='Name of yaml file in root directory. E.g. sample_yaml.yml')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    with open(args.config_filename) as f:
        config = yaml.safe_load(f)

    generate_pipeline(config)
    shutil.copyfile(args.config_filename, os.path.join('experiment_runs', config['experiment_name'], f"{config['experiment_name']}.yml"))

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
