import os
import argparse

'''
A R G U M E N T S
P A R S I N G
'''
parser = argparse.ArgumentParser(description='Detailed options for momentum config.')
parser.add_argument('--configuration', type=str, default="realistic_scenario",
                    help='name of the configuration'
                         'inside custom_configs/')
parser.add_argument('--scale', type=float, default=1,
                    help='scale of the simulation')
parser.add_argument('--num_simulations', type=int, default=3,
                    help='number of parallel simulations to generate')
args = parser.parse_args()

configuration, scale, num_simulations = args.configuration, \
                                        args.scale, \
                                        args.num_simulations

'''
S I M U L A T I O N S
'''
print(f"Running {num_simulations} parallel simulations with configuration {configuration} and {scale} scale")

if num_simulations == 1:
    # run a single simulation
    os.system(f"python run_simulation.py --configuration {configuration} --scale {scale}")
else:
    # run multiple simulations in parallel
    os.system(" & ".join(
        [f"python run_simulation.py --configuration {configuration} --scale {scale}" for _ in range(num_simulations)]))
