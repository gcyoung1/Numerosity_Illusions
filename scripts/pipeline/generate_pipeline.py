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


def create_gen_stimuli_command(dataset_object, experiment_name):    
    if 'nasr' in dataset_object['dataset_name']:
        command = f"ml matlab/R2022b;matlab -nodisplay -nosplash -nodesktop -r \"experiment_name='{experiment_name}';number_sets=[{','.join([str(x) for x in dataset_object['numerosities']])}];num_pics_per_category={dataset_object['num_pics_per_category']};cd('scripts/stimuli');Stimulus_generation_Nasr(experiment_name,number_sets,num_pics_per_category); exit;\""
    elif 'dewind' in dataset_object['dataset_name']:
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
        command += f" --experiment_name {experiment_name}"

    else:
        raise NotImplementedError(f"Dataset {dataset_object['dataset_name']} not supported. Make sure you prepend the dataset type in the dataset name.")

    return command
    

def create_gen_stimuli_sbatch(config):
    hours = 5
    
    commands = ""
    for numerosity_neurons_dataset in config['numerosity_neurons_datasets']:
        numerosity_neurons_command = create_gen_stimuli_command(numerosity_neurons_dataset, config['experiment_name'])
        commands += numerosity_neurons_command + "\n\n"
    for activations_dataset in config['activations_datasets']:
        activations_dataset_command = create_gen_stimuli_command(activations_dataset, config['experiment_name'])
        commands += activations_dataset_command + "\n\n"

    header = f"""#!/bin/bash
#
#SBATCH --job-name=1_gen_stimuli_{config['experiment_name']}
#SBATCH --output=jobs/1_gen_stimuli_{config['experiment_name']}_%j.out
#
#SBATCH --time={hours}:00:00

source activate numerosity_illusions;

""" 


    return header + commands

def create_save_layers_sbatch(config):
    dataset_names = [dataset['dataset_name'] for dataset in config['activations_datasets']] + [dataset['dataset_name'] for dataset in config['numerosity_neurons_datasets']]

    hours = int(len(dataset_names) * len(config['models']) * 0.2) + 1 # Empirically determined
    header = f"""#!/bin/bash
#
#SBATCH --job-name=2_save_layers_{config['experiment_name']}
#SBATCH --output=jobs/2_save_layers_{config['experiment_name']}_%j.txt
#
#SBATCH --time={hours}:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH -p gpu
#SBATCH -G 2

source activate numerosity_illusions;

"""
    for model_object in config['models']:
        model_command = f"echo {model_object['model_name']}; for dataset in {' '.join(dataset_names)}; do echo $dataset; python -m scripts.models.save_layers --model_name {model_object['model_name']} --dataset_name $dataset --layers {' '.join(model_object['layers'])}  --num_workers 1 --experiment_name {config['experiment_name']}"
        if config['downsample_layers']:
            model_command += f" --downsample_layers --num_kept_neurons {config['num_kept_neurons']}" 
        if model_object['model_name'] in ['vit_pretrained', 'vit_random']:
            model_command += f" --batch_size 20"
        model_command += "; done \n\n"
        header += model_command
    return header

def estimate_seconds_per_neuron(dataset_object):
    if 'dewind' in dataset_object['dataset_name']:
        num_images = dataset_object["num_pics_per_category"]*len(dataset_object["sizes"])*len(dataset_object["spacings"])*len(dataset_object["numerosities"])
        num_levels = max(len(dataset_object["sizes"]), len(dataset_object["spacings"]), len(dataset_object["numerosities"]))
        seconds_per_neuron = (0.55*(0.08/1080)*num_images + 0.4)*2.5**(max(num_levels-6,0)) # Empirically determined
    elif 'nasr' in dataset_object['dataset_name']:
        num_images = dataset_object["num_pics_per_category"]*len(dataset_object["numerosities"])*3
        num_levels = 3
        seconds_per_neuron = (0.55*(0.08/1080)*num_images) # Empirically determined
   
    
    return seconds_per_neuron

def create_identify_numerosity_neurons_sbatch(config):
    commands = ""
    seconds = 0
    for model_object in config['models']:
        model_name = model_object['model_name']
        if not config['downsample_layers']:
            layer_size_dict = model_utils.get_layer_size_dict(model_name)
        for layer in model_object['layers']:
            layer_size = layer_size_dict[layer] if not config['downsample_layers'] else config['num_kept_neurons']
            for dataset_object in config['numerosity_neurons_datasets']:
                seconds_per_neuron = estimate_seconds_per_neuron(dataset_object)
                seconds += seconds_per_neuron * layer_size
                dataset_name = dataset_object['dataset_name']
                for selection_method in config['selection_methods']:
                    commands += f"echo {model_name}; echo {layer}; echo {dataset_name}; echo {selection_method}; python -m scripts.analysis.identify_numerosity_neurons --model_directory {model_name} --dataset_name {dataset_name} --layer {layer} --selection_method {selection_method} --experiment_name {config['experiment_name']};\n\n"

    hours = int(seconds // 3600)
    hourbuffer = int(hours * 0.2) + 1
    hours += hourbuffer

    header = f"""#!/bin/bash
#
#SBATCH --job-name=3_identify_numerosity_neurons_{config['experiment_name']}
#SBATCH --output=jobs/3_identify_numerosity_neurons_{config['experiment_name']}_%j.txt
#
#SBATCH --time={hours}:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem 80G

source activate numerosity_illusions;

"""

    return header + commands

def create_save_tuning_sbatch(config):
    
    commands = ""

    for plot_object in config['plots']:
        for activations_dataset_name in plot_object['activations_dataset_names']:
            for model_object in config['models']:
                model_name = model_object['model_name']
                for layer in model_object['layers']:
                    for selection_method in config['selection_methods']:
                        command = f"echo {model_name}; echo {activations_dataset_name}; echo {layer}; python -m scripts.analysis.save_tuning_curves --model_directory {model_name} --layer {layer} --numerosity_neurons_dataset_name {plot_object['numerosity_neurons_dataset_name']} --selection_method {selection_method} --activations_dataset_name {activations_dataset_name} --experiment_name {config['experiment_name']};\n\n"
                        commands += command

    header = f"""#!/bin/bash
#
#SBATCH --job-name=4_save_tuning_curves_{config['experiment_name']}
#SBATCH --output=jobs/4_save_tuning_curves_{config['experiment_name']}_%j.txt
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

source activate numerosity_illusions


"""
    return header + commands

def create_plot_tuning_sbatch(config):
    commands = ""

    for plot_object in config['plots']:
        activations_dataset_names = ",".join(plot_object['activations_dataset_names'])
        for model_object in config['models']:
            model_name = model_object['model_name']
            for layer in model_object['layers']:
                for selection_method in config['selection_methods']:
                    command = f"echo {model_name}; echo {activations_dataset_names}; echo {layer}; python -m scripts.plotting.plot_tuning_curves --model_directory {model_name} --layer {layer} --numerosity_neurons_dataset_name {plot_object['numerosity_neurons_dataset_name']} --selection_method {selection_method} --experiment_name {config['experiment_name']}"
                    if activations_dataset_names:
                        command += "--activations_dataset_name {activations_dataset_names}"
                    command += ";\n\n"
                    commands += command


    header = f"""#!/bin/bash
#
#SBATCH --job-name=5_plot_tuning_curves_{config['experiment_name']}
#SBATCH --output=jobs/5_plot_tuning_curves_{config['experiment_name']}_%j.txt
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

source activate numerosity_illusions


"""
    return header + commands

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
