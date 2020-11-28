import os
import re
from os import mkdir, listdir
from os.path import exists, join, isfile

# functions from ABIDES - ignore it
from abides.realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF


def create_orderbooks(exchange_path, ob_path):
    """ Creates orderbook DataFrames from ABIDES exchange output file and orderbook output file. """

    processed_orderbook = make_orderbook_for_analysis(exchange_path, ob_path, num_levels=1,
                                                      hide_liquidity_collapse=False)
    cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                            (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
    transacted_orders = cleaned_orderbook.loc[cleaned_orderbook.TYPE == "ORDER_EXECUTED"]
    transacted_orders['SIZE'] = transacted_orders['SIZE'] / 2

    return processed_orderbook, transacted_orders, cleaned_orderbook



# parses console arguments
import argparse
import sys

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('-n', '--simulation_name', type=str, default=None,
                    help='Log directory path of the ABIDES simulation to convert to .csv')
parser.add_argument('--config_help',
                    help='Print this help message and exit')

args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  sys.exit()

simulation_name_param = args.simulation_name

# our paths
abides_log_path, data_path = "abides/log", "data"
# eventually creates the data folder
if not exists(data_path):
    mkdir(data_path)

# collects the names of the simulations we want to parse
simulations_names = []
if simulation_name_param:
    # raises an error if the simulation does not exists
    if not exists(join(abides_log_path, simulation_name_param)):
        err_string = f""
        err_string += f"Simulation with name {simulation_name_param} not present in {abides_log_path}\n"
        err_string += f"Available simulations names:\n"
        for simulation_name in [dir for dir in listdir(abides_log_path) if not isfile(dir)]:
            err_string += f"\t{simulation_name}\n"
        err_string = err_string.strip()
        raise Exception(err_string)
    # if we have a particular simulation with -n, use only it
    simulations_names += [simulation_name_param]
else:
    # else search for all the simulation
    simulations_names += [dir for dir in listdir(abides_log_path) if not isfile(dir)]

for simulation_name in simulations_names:
    # appends the prefix to simulation's path
    abides_simulation_path, data_simulation_path = join(abides_log_path, simulation_name), \
                                                   join(data_path, simulation_name)
    # eventually creates the simulation folder into the data folder
    if not exists(data_simulation_path):
        os.mkdir(data_simulation_path)
    # scans for the orderbooks and agents .bz2 files, saving them, for example:
    # agent_filename = "ExchangeAgent0.bz2"
    # orderbook_filenames = ["ORDERBOOK_JPM_FULL.bz2"]
    agent_filename, orderbook_filenames = None, []
    for file in os.listdir(abides_simulation_path):
        if re.fullmatch(r"ORDERBOOK_.*\.bz2", file):
            orderbook_filenames += [file]
        if not agent_filename and re.fullmatch(r"ExchangeAgent.*\.bz2", file):
            agent_filename = file
    # saves the .csv into data/{simulation_name}
    for orderbook_filename in orderbook_filenames:
        # uncompresses the pickles
        processed_orderbook, transacted_orders, cleaned_orderbook = create_orderbooks(
            join(abides_simulation_path, agent_filename),
            join(abides_simulation_path, orderbook_filename))
        # saves .csv to files
        orderbook_file_wo_extension = os.path.splitext(orderbook_filename)[0]
        processed_orderbook.to_csv(
            join(data_path, simulation_name, orderbook_file_wo_extension) + "_processed_orderbook" + ".csv",
            index=True)
        transacted_orders.to_csv(
            join(data_path, simulation_name, orderbook_file_wo_extension) + "_transacted_orders" + ".csv",
            index=True)
        cleaned_orderbook.to_csv(
            join(data_path, simulation_name, orderbook_file_wo_extension) + "_cleaned_orderbook" + ".csv",
            index=True)
