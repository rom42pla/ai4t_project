import argparse
import numpy as np

'''
A R G U M E N T S
P A R S I N G
'''
parser = argparse.ArgumentParser(description='Run an ABIDES simulation')
parser.add_argument('--configuration', type=str, default="realistic_scenario",
                    help='name of the configuration to use, '
                         'inside custom_configs/')
parser.add_argument('--plot_config', type=str, default="plot_realistic_scenario",
                    help='name of the .json with the parameters of the plot, '
                         'inside abides/util/plotting/configs/')
parser.add_argument('--seed', type=int, default=np.random.randint(100, 999999),
                    help='seed of the simulation')
parser.add_argument('--scale', type=float, default=1,
                    help='scale of the simulation (proportion of agents wrt full simulation)')
parser.add_argument('--hours', type=float, default=8,
                    help='hours of simulation to reproduce, '
                         'starting always from 09:00AM up to 17:00PM')

parser.add_argument('--num_impacts', type=int, default=1,
                    help='number of impacts on the ETF, '
                         'equally distributed during the simulation')
parser.add_argument('--impacts_greed', type=float, default=0.5,
                    help='percentage of money used by impact agents')
args = parser.parse_args()

# retrieves the arguments
configuration, plot_config, seed, scale, hours = args.configuration, \
                                                 args.plot_config, \
                                                 args.seed, \
                                                 args.scale, \
                                                 args.hours
assert 1 <= hours <= 8

num_impacts, impacts_greed = args.num_impacts, \
                             args.impacts_greed
assert num_impacts >= 0
assert 0 < impacts_greed <= 1

'''
S I M U L A T I O N
'''
import os
from os import mkdir
from os.path import join, exists
import json
import pandas as pd

print(f"Running simulation {configuration} with seed {seed} and {scale} scale for {hours} hours")

# updates our custom configurations into ABIDES' folder
os.system("python update_custom_configs.py")

# launches the simulation
os.system(f"cd ../abides;"
          f"python -u abides.py -c {configuration} -l {configuration} -b 0 -s {seed} -o True -sc {scale} --hours {hours} "
          f"--num_impacts {num_impacts} --impacts_greed {impacts_greed}")

# prepares the data folders
data_path = join("..", "data")
if not exists(data_path):
    mkdir(data_path)
simulation_path = join(data_path, f"{configuration}{seed}")
simulation_plots_path = join(simulation_path, f"plots")
if not exists(simulation_path):
    mkdir(simulation_path)
    mkdir(simulation_plots_path)

# .json template for the plots
secondary_market_open = pd.to_timedelta("09:00:00")
secondary_market_close = secondary_market_open + pd.Timedelta(hours, unit="hours")
plot_template = {
    "xmin": str(secondary_market_open + pd.Timedelta(1, unit="minutes")).split()[-1],
    "xmax": str(secondary_market_close).split()[-1],
    "linewidth": 0.7,
    "no_bids_color": "blue",
    "no_asks_color": "red",
    "transacted_volume_binwidth": 30,
    "shade_start_time": "01:00:00",
    "shade_end_time": "01:30:00"
}
simulation_plots_config_path = join(simulation_plots_path, "plots_config.json")
with open(simulation_plots_config_path, "w") as fp:
    json.dump(plot_template, fp=fp, indent=4)

# plots the charts
os.system(f"cd ../abides/util/plotting/;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_ETF_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/ETF_plot.png -c ../../../data/{configuration}{seed}/plots/plots_config.json;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_SYM1_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/SYM1_plot.png -c ../../../data/{configuration}{seed}/plots/plots_config.json;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_SYM2_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/SYM2_plot.png -c ../../../data/{configuration}{seed}/plots/plots_config.json;"
          f"python -u liquidity_telemetry.py ../../log/{configuration}{seed}/ExchangeAgent0.bz2 ../../log/{configuration}{seed}/ORDERBOOK_SYM3_FULL.bz2 -o ../../../data/{configuration}{seed}/plots/SYM3_plot.png -c ../../../data/{configuration}{seed}/plots/plots_config.json;")

# eventually visualizes the charts in an app
os.system(f"cd ..;"
          f"fim data/{configuration}{seed}/plots/;")

# gets .csv of the simulation
os.system(f"cd ..; ls; python get_csv_simulations.py --simulation_name {configuration}{seed}")

print(f"Ended simulation {configuration} with seed {seed} and {scale} scale for {hours} hours")
