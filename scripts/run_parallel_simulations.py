import os
import argparse

'''
A R G U M E N T S
P A R S I N G
'''
parser = argparse.ArgumentParser(description='Run multiple ABIDES simulations in parallel')
parser.add_argument('--configuration', type=str, default="realistic_scenario",
                    help='name of the configuration to use, '
                         'inside custom_configs/')
parser.add_argument('--scale', type=float, default=1,
                    help='scale of the simulation')
parser.add_argument('--hours', type=float, default=1,
                    help='hours of simulation to reproduce, '
                         'starting always from 09:00AM up to 17:00PM')
parser.add_argument('--num_simulations', type=int, default=3,
                    help='number of parallel simulations to generate')

parser.add_argument('--num_impacts', type=int, default=1,
                    help='number of impacts on the ETF, '
                         'equally distributed during the simulation')
parser.add_argument('--impacts_greed', type=float, default=0.5,
                    help='percentage of money used by impact agents')
args = parser.parse_args()

configuration, scale, hours, num_simulations = args.configuration, \
                                               args.scale, \
                                               args.hours, \
                                               args.num_simulations

assert 1 <= hours <= 8

num_impacts, impacts_greed = args.num_impacts, \
                             args.impacts_greed
assert num_impacts >= 0
assert 0 < impacts_greed <= 1

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
        [f"python run_simulation.py --configuration {configuration} --scale {scale} "
         f"--num_impacts {num_impacts} --impacts_greed {impacts_greed}"
         for _ in range(num_simulations)]))
