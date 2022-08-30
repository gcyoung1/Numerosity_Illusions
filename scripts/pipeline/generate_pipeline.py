import yaml
import argparse
import shutil
import os
import time

def create_gen_stimuli_command(dataset_object):
    base = "python -m scripts.stimuli.gen_dewind_circles"
    if dataset_object['interpolate']:
        for param in ['sizes', 'spacings', 'numerosities']:
            assert len(dataset_object[param]) == 2
            start,end = dataset_object[param]
            assert end >= start
            step_size = (end-start)/dataset_object['num_steps']
            dataset_object[param] = " ".join([str(start + i*step_size) for i in range(dataset_object['num_steps']+1)])
    binary_args = ["hollow", "illusory", "linear_args"]
    other_args = ["interpolate", "num_steps"]
    command = base + "".join([f" --{arg_name} {arg_value}" for arg_name, arg_value in dataset_object.items() if arg_name not in other_args+binary_args])
    for binary_arg in binary_args:
        if dataset_object.get(binary_arg, False):
            command += f" --{binary_arg}"
    return command
    

def create_gen_stimuli_sbatch(config):
    header = f"""#!/bin/bash
#
#SBATCH --job-name=gen_stimuli_{config['experiment_name']}
#SBATCH --output=jobs/gen_stimuli_{config['experiment_name']}_%j.out
#
#SBATCH --time=3:00:00

"""
    numerosity_neurons_command = create_gen_stimuli_command(config['numerosity_neurons_dataset'])
    sbatch = header + numerosity_neurons_command + "\n\n"
    for activations_dataset in config['activations_datasets']:
        activations_dataset_command = create_gen_stimuli_command(activations_dataset)
        sbatch += activations_dataset_command + "\n\n"
    return sbatch

def create_save_layers_sbatch(config):
    dataset_names = [dataset['dataset_name'] for dataset in config['activations_datasets']] + [config['numerosity_neurons_dataset']['dataset_name']]
    time = int(len(dataset_names) * 0.5 * 2) + 1
    sbatch = f"""#!/bin/bash
#
#SBATCH --job-name=save_layers_{config['experiment_name']}
#SBATCH --output=jobs/save_layers_{config['experiment_name']}_%j.txt
#
#SBATCH --time={time}:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH -p gpu
#SBATCH -G 2

for model in {config["model_name"]}_random {config["model_name"]}_pretrained;
do echo $model;
for dataset in {" ".join(dataset_names)};
do echo $dataset; 
python -m scripts.models.save_layers --model $model --stimulus_directory $stim_dir --layers {" ".join(config['layers'])}  --num_workers 1;
done;
done
"""
    return sbatch


def create_save_tuning_sbatch(config):
    activations_dataset_names = [dataset['dataset_name'] for dataset in config['activations_datasets']] + [config['numerosity_neurons_dataset']['dataset_name']]
    time = int(len(activations_dataset_names) * 0.5 * 2) + 1
    sbatch = f"""#!/bin/bash
#
#SBATCH --job-name=save_tuning_curves_{config['experiment_name']}
#SBATCH --output=jobs/save_tuning_curves_{config['experiment_name']}_%j.txt
#
#SBATCH --time={time}:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem 192G

source activate numerosity_illusions

for model in {config["model_name"]}_random {config["model_name"]}_pretrained;
do echo $model;
for dataset in {" ".join(activations_dataset_names)};
do echo $dataset; 
for layer in {" ".join(config['layers'])};
do echo $layer;
python -m scripts.analysis.save_tuning_curves --model_directory $model --layer $layer --numerosity_neurons_dataset_directory {config['numerosity_neurons_dataset']['dataset_name']} --selection_method {config['selection_method']} --activations_dataset_directory $dataset;
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
#SBATCH --job-name=plot_tuning_curves_{config['experiment_name']}
#SBATCH --output=jobs/plot_tuning_curves_{config['experiment_name']}_%j.txt
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

source activate numerosity_illusions

for model in {config["model_name"]}_random {config["model_name"]}_pretrained;
do echo $model;
for dataset in {" ".join(activations_dataset_names)};
do echo $dataset; 
for layer in {" ".join(config['layers'])};
do echo $layer;
python -m scripts.analysis.plot_tuning_curves --model_directory $model --layer $layer --numerosity_neurons_dataset_directory {config['numerosity_neurons_dataset']['dataset_name']} --selection_method {config['selection_method']} --activations_dataset_directory $dataset;
done;
done;
done
"""
    return sbatch


def create_scheduler_sh(config):
    sh = f"""#!/bin/bash

ID=$(sbatch --parsable experiment_runs/{config['experiment_name']}/submit_gen_stimuli.sbatch)
shift 
for script in submit_save_layers.sbatch submit_save_tuning.sbatch submit_plot_tuning.sbatch; do
  ID=$(sbatch --parsable --dependency=afterok:$ID experiment_runs/{config['experiment_name']}/$script)
done

"""
    return sh

def generate_pipeline(config):
    # Create experiment folders
    for location in ['data/stimuli', 'data/models', 'experiment_runs']:
        os.mkdir(os.path.join(location, config['experiment_name']))
    # Create folder to hold sbatch files
    pipeline_folder = os.path.join('experiment_runs', config['experiment_name'])
    # Create sbatch files for each stage of experiment
    gen_stimuli_sbatch = create_gen_stimuli_sbatch(config)
    with open(os.path.join(pipeline_folder, "submit_gen_stimuli.sbatch"),"w") as f:
        f.write(gen_stimuli_sbatch)
    save_layers_sbatch = create_save_layers_sbatch(config)
    with open(os.path.join(pipeline_folder, "submit_save_layers.sbatch"),"w") as f:
        f.write(save_layers_sbatch)
    save_tuning_sbatch = create_save_tuning_sbatch(config)
    with open(os.path.join(pipeline_folder, "submit_save_tuning.sbatch"),"w") as f:
        f.write(save_tuning_sbatch)
    plot_tuning_sbatch = create_plot_tuning_sbatch(config)
    with open(os.path.join(pipeline_folder, "submit_plot_tuning.sbatch"),"w") as f:
        f.write(plot_tuning_sbatch)
    # Create sh file to schedule sbatch files
    scheduler_sh = create_scheduler_sh(config)
    with open(os.path.join(pipeline_folder, "scheduler.sh"),"w") as f:
        f.write(scheduler_sh)

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
