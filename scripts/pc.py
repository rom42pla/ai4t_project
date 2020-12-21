import time
import json
from pprint import pprint
from os import listdir
from os.path import join
from itertools import combinations

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import argparse

parser = argparse.ArgumentParser(description='Detailed options for PC')
parser.add_argument('--simulation_name', default="realistic_scenario654321",
                    help='Seed for the random generator')
parser.add_argument('--seed', default=123456, type=int,
                    help='Seed for the random generator')
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)


def rolling_mean(df, columns=[], seconds=10):
    if isinstance(columns, str):
        columns = [columns]
    if len(columns) == 0:
        columns = list(df.columns)
    df = df.copy().loc[:, columns]

    out = pd.DataFrame(columns=columns)
    for i, g in df.groupby([(pd.to_datetime(df.index) - pd.to_datetime(df.index)[0])
                                    .astype(f'timedelta64[{seconds}s]')]):
        out = out.append(g.mean(), ignore_index=True)
    return out


data_path = join("..", "data")
simulation_path = join(data_path, args.simulation_name)

symbols_names = ["ETF", "SYM1", "SYM2", "SYM3"]
rows_to_drop = 5000
mid_prices, transacted_volumes = {}, {}
print(f"Computing rolling mean to reduce instants of time...")
starting_time = time.time()
for symbol_name in symbols_names:
    orderbook, orderbook_transacted = pd.read_csv(join(simulation_path,
                                                       f"ORDERBOOK_{symbol_name}_FULL_processed_orderbook.csv")) \
                                          .rename(columns={'index': 'time'}).set_index("time"), \
                                      pd.read_csv(join(simulation_path,
                                                       f"ORDERBOOK_{symbol_name}_FULL_transacted_orders.csv")) \
                                          .rename(columns={'index': 'time'}).set_index("time")
    orderbook.index, orderbook_transacted.index = pd.to_datetime(orderbook.index), \
                                                  pd.to_datetime(orderbook_transacted.index)

    mid_prices[symbol_name], transacted_volumes[symbol_name] = rolling_mean(orderbook, columns="MID_PRICE")[
                                                                   "MID_PRICE"].values, \
                                                               rolling_mean(orderbook, columns="SIZE")["SIZE"].values
print(f"Done in {time.time() - starting_time}")

for symbol_name in symbols_names:
    fig, ax = plt.subplots(2)
    fig.suptitle(f'Plots for symbol {symbol_name}')
    mid_price_ax = sns.lineplot(data=mid_prices[symbol_name],
                                ax=ax[0])
    mid_price_ax.set(xticklabels=[], xlabel="Time", ylabel="Mid price")

    size_ax = sns.lineplot(data=transacted_volumes[symbol_name],
                           ax=ax[1])
    size_ax.set(xticklabels=[], xlabel="Time", ylabel="Transacted volume")
plt.tight_layout()
plt.plot()


def get_manifold(*time_series, plot=True):
    # eventually unpacks the list
    if len(time_series) == 1:
        time_series = time_series[0]
    # checks that there are at least two symbols to compare
    assert len(time_series) > 1
    for i_symbol, time_serie in enumerate(time_series):
        # checks that the input is a valid numpy array
        assert isinstance(time_serie, np.ndarray)
        # checks that the shapes of each symbol must match with the others
        assert len(time_serie.shape) == 1
        if i_symbol > 0:
            assert time_series[i_symbol].shape == time_series[i_symbol - 1].shape
    manifold = np.stack(time_series, axis=1)

    # eventually plot the manifold
    if plot:
        dimensions = manifold.shape[-1]
        # 2d plots are not working
        # if dimensions == 2:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.plot(xs=time_series[0], ys=time_series[1])
        #     plt.show()
        if dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(xs=manifold[:, 0], ys=manifold[:, 1], zs=manifold[:, 2])
            plt.show()
    return manifold


def get_nn_and_distances(manifold, num_nn=5):
    # distances = distance_matrix(manifold, manifold, p=1)
    nn_distances, nn_indices = NearestNeighbors(n_neighbors=num_nn + 1, metric="manhattan", algorithm="auto") \
        .fit(manifold).kneighbors(manifold)
    nn_distances, nn_indices = nn_distances[:, 1:], nn_indices[:, 1:]

    w = np.exp(-nn_distances) / \
        (np.sum(np.exp(-nn_distances)))

    return w, nn_indices


def signature(a):
    a = a.copy()
    a[a > 0] = 1
    a[a < 0] = -1
    return a


def PC(*time_series, E=3, tau=1):
    # eventually unpacks the list
    if len(time_series) == 1:
        time_series = time_series[0]
    # checks that there are at least two symbols to compare
    assert len(time_series) > 1
    for i_symbol, time_serie in enumerate(time_series):
        # checks that the input is a valid numpy array
        assert isinstance(time_serie, np.ndarray)
        # checks that the shapes of each symbol must match with the others
        assert len(time_serie.shape) == 1
        if i_symbol > 0:
            assert time_series[i_symbol].shape == time_series[i_symbol - 1].shape

    attractor = get_manifold(time_series)
    manifolds = [get_manifold([time_serie[e * tau:-((E - e - 1) * tau) if E - e - 1 != 0 else None]
                               for e in range(E)], plot=False)
                 for time_serie in time_series]
    s = [(manifold[:, :-1] - manifold[:, 1:]) / manifold[:, 1:]
         for manifold in manifolds]

    w, nn_indices = [], []
    for manifold in manifolds:
        nn = get_nn_and_distances(manifold=manifold, num_nn=5)
        w += [nn[0]]
        nn_indices += [nn[1]]

    S = [np.sum(np.array([w[i], w[i]]).transpose((1, 2, 0)) * s[i][nn_indices[i]], axis=1)
         for i in range(len(manifolds))]

    P = [signature(S[i]) for i in range(len(manifolds))]

    causalities_matrix = np.zeros(shape=(len(manifolds), len(manifolds)))
    for i1, i2 in list(combinations(range(len(manifolds)), 2)):
        pattern_causalities = np.zeros(shape=P[0].shape[0])
        for i_pattern_causality, (signature1, signature2) in enumerate(list(zip(P[i1], P[i2]))):
            product = np.dot(signature1, signature2)
            if not (np.isfinite(product)):
                print(signature1, signature2)
            pattern_causalities[i_pattern_causality] = ((E - 1) / product) if np.isfinite(
                product) and product != 0 else 0
        causality_factor = pattern_causalities.mean()
        causalities_matrix[i1, i2] = causality_factor
        causalities_matrix[i2, i1] = causality_factor
    return causalities_matrix


print(f"Computing PC between couple of symbols...")
starting_time = time.time()
# computes PC between the symbols
causalities = PC([mid_prices[symbol_name] for symbol_name in symbols_names])

# outputs the results
causalities_json = {}
for i_symbol1, i_symbol2 in list(combinations(range(len(causalities)), 2)):
    symbol_name1, symbol_name2 = symbols_names[i_symbol1], symbols_names[i_symbol2]
    causality_strength = causalities[i_symbol1][i_symbol2]
    if not symbol_name1 in causalities_json:
        causalities_json[symbol_name1] = {}
    if not symbol_name2 in causalities_json:
        causalities_json[symbol_name2] = {}
    causalities_json[symbol_name1][symbol_name2] = causality_strength
    causalities_json[symbol_name2][symbol_name1] = causality_strength
# saves into a file
with open(join(simulation_path, "pc.json"), "w") as fp:
    json.dump(causalities_json, fp, indent=4)
pprint(causalities_json)
print(f"Done in {time.time() - starting_time}")

