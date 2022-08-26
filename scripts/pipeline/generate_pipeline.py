import yaml
import argparse


import os
import time


# def generate_stimuli_sbatch():

# def generate_save_layers_sbatch():

# def generate_save_tuning_sbatch():

# def generate_plot_tuning_sbatch():

# def generate_executive_sh():

# def generate_pipeline():

if __name__ == '__main__':
    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Generate pipeline for running experiment.')
    parser.add_argument('config_file', type=str, 
                        help='Name of yaml file in root directory. E.g. sample_yaml.yml')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    print(config)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
