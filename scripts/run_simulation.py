import os
import argparse
import numpy as np

'''
A R G U M E N T S
P A R S I N G
'''
parser = argparse.ArgumentParser(description='Run an ABIDES simulation')
parser.add_argument('--configuration', type=str, default="realistic_scenario",
                    help='name of the configuration to use,'
                         'inside custom_configs/')
parser.add_argument('--plot_config', type=str, default="plot_realistic_scenario",
                    help='name of the .json with the parameters of the plot,'
                         'inside abides/util/plotting/configs/')
parser.add_argument('--seed', type=int, default=np.random.randint(100, 999999),
                    help='seed of the simulation')
parser.add_argument('--scale', type=float, default=1,
                    help='scale of the simulation (proportion of agents wrt full simulation)')
parser.add_argument('--hours', type=float, default=8,
                    help='hours of simulation to reproduce,'
                         'starting always from 09:00AM up to 17:00PM')
args = parser.parse_args()

# retrieves the arguments
configuration, plot_config, seed, scale, hours = args.configuration, \
                                                 args.plot_config, \
                                                 args.seed, \
                                                 args.scale, \
                                                 args.hours
assert 1 <= hours <= 8

'''
S I M U L A T I O N
'''
print(f"Running simulation {configuration} with seed {seed} and {scale} scale for {hours} hours")

# updates our custom configurations into ABIDES' folder
os.system("python update_custom_configs.py")

# launches the simulation
os.system(f"cd ../abides;"
          f"python -u abides.py -c {configuration} -l {configuration} -b 0 -s {seed} -o True -sc {scale} --hours {hours}")

# plots the charts
os.system(f"cd ..;"
          f"mkdir data;"
          f"mkdir data/{configuration}{seed};"
          f"mkdir data/{configuration}{seed}/plots;"
          f"cd abides/util/plotting/;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_ETF_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/ETF_plot.png -c configs/{plot_config}.json;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_SYM1_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/SYM1_plot.png -c configs/{plot_config}.json;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_SYM2_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/SYM2_plot.png -c configs/{plot_config}.json;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_SYM3_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/SYM3_plot.png -c configs/{plot_config}.json;")

# eventually visualizes the charts in an app
os.system(f"cd ..;"
          f"fim data/{configuration}{seed}/plots/;")
