import pandas as pd
import sys
import os

sys.path.append('../..')

from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta, datetime
import argparse
import json
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000

PLOT_PARAMS_DICT = None

LIQUIDITY_DROPOUT_BUFFER = 360  # Time in seconds used to "buffer" as indicating start and end of trading


def create_orderbooks(exchange_path, ob_path):
    """ Creates orderbook DataFrames from ABIDES exchange output file and orderbook output file. """

    print("Constructing orderbook...")
    processed_orderbook = make_orderbook_for_analysis(exchange_path, ob_path, num_levels=1,
                                                      hide_liquidity_collapse=False)
    cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                            (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
    transacted_orders = cleaned_orderbook.loc[cleaned_orderbook.TYPE == "ORDER_EXECUTED"]
    transacted_orders['SIZE'] = transacted_orders['SIZE'] / 2

    return processed_orderbook, transacted_orders, cleaned_orderbook


def main(exchange_paths, ob_paths, title=None, outfile='liquidity_telemetry.png', verbose=False):
    """ Processes orderbook from files, creates the liquidity telemetry plot and (optionally) prints statistics. """

    all_mid_prices = []
    for i in range(len(ob_paths)):
        processed_orderbook, transacted_orders, cleaned_orderbook = create_orderbooks(exchange_paths[i], ob_paths[i])
        all_mid_prices.append(cleaned_orderbook["MID_PRICE"])
        
    #  preamble
    fig, axes = plt.subplots()
    fig.set_size_inches(h=23, w=15)

    date = all_mid_prices[0].index[0].date()
    midnight = pd.Timestamp(date)
    xmin = midnight + pd.to_timedelta(PLOT_PARAMS_DICT['xmin'])
    xmax = midnight + pd.to_timedelta(PLOT_PARAMS_DICT['xmax'])
    shade_start = midnight + pd.to_timedelta(PLOT_PARAMS_DICT['shade_start_time'])
    shade_end = midnight + pd.to_timedelta(PLOT_PARAMS_DICT['shade_end_time'])

    #  top plot -- mid price + fundamental
    y = None
    for el in all_mid_prices:
        cut_el = el.loc[xmin:xmax]
        if y is None:
            y = np.asarray([cut_el[i] for i in range(len(cut_el))])[:11000]
        else:
            y += np.asarray([cut_el[i] for i in range(len(cut_el))])[:11000]
    y = y / len(all_mid_prices)

    plt.plot(np.asarray(list(range(len(y)))), y, color='black', label="Mid price")

    if title:
        plt.suptitle(title, fontsize=18, y=0.905)

    plt.subplots_adjust(hspace=0.05)
    fig.savefig(outfile, format='png', dpi=300, transparent=False, bbox_inches='tight',
                pad_inches=0.03)


def check_str_png(s):
    """ Check if string has .png extension. """
    if not isinstance(s, str):
        raise TypeError("Input must be of type str")
    if not s.endswith('.png'):
        raise ValueError("String must end with .png")
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI utility for inspecting liquidity issues and transacted volumes '
                                                 'for a day of trading.')

    parser.add_argument("-s", dest='stream', type=str, action="append", help='ABIDES order stream in bz2 format. '
                                                 'Typical example is `ExchangeAgent.bz2`')
    parser.add_argument('-b', dest='book', type=str, action="append",
                        help='ABIDES order book output in bz2 format. Typical example is '
                                               'ORDERBOOK_TICKER_FULL.bz2')
    parser.add_argument('-o', '--out_file',
                        help='Path to png output file. Must have .png file extension',
                        type=check_str_png,
                        default='liquidity_telemetry.png')
    parser.add_argument('-t', '--plot-title',
                        help="Title for plot",
                        type=str,
                        default=None
                        )
    parser.add_argument('-v', '--verbose',
                        help="Print some summary statistics to stderr.",
                        action='store_true')
    parser.add_argument('-c', '--plot-config',
                        help='Name of config file to execute. '
                             'See configs/telemetry_config.example.json for an example.',
                        default='configs/telemetry_config.example.json',
                        type=str)

    args, remaining_args = parser.parse_known_args()

    out_filepath = args.out_file
    stream = args.stream
    book = args.book
    title = args.plot_title
    verbose = args.verbose
    with open(args.plot_config, 'r') as f:
        PLOT_PARAMS_DICT = json.load(f)

    main(stream, book, title=title, outfile=out_filepath, verbose=verbose)
