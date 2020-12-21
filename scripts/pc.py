from pprint import pprint
from os import listdir
from os.path import join

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

data_path = join("..", "data")
sample_simulation_path = join(data_path, "realistic_scenario123456")

symbols_names = ["ETF", "SYM1", "SYM2", "SYM3"]
rows_to_drop = 5000
mid_prices, transacted_volumes = {}, {}
for symbol_name in symbols_names:
    orderbook, orderbook_transacted = pd.read_csv(
        join(sample_simulation_path, f"ORDERBOOK_{symbol_name}_FULL_processed_orderbook.csv")).rename(
        columns={'index': 'time'}), \
                                      pd.read_csv(join(sample_simulation_path,
                                                       f"ORDERBOOK_{symbol_name}_FULL_transacted_orders.csv")).rename(
                                          columns={'index': 'time'})
    mid_prices[symbol_name], transacted_volumes[symbol_name] = orderbook["MID_PRICE"].values[rows_to_drop:], \
                                                               orderbook_transacted["SIZE"].values[rows_to_drop:]

sample_indexes = np.linspace(start=0, stop=mid_prices["ETF"].shape[0] - rows_to_drop, num=1000,
                             endpoint=False, dtype=np.int)
sample_indexes_transacted = np.linspace(start=0, stop=transacted_volumes["ETF"].shape[0] - rows_to_drop, num=1000,
                                        endpoint=False, dtype=np.int)

for symbol_name in symbols_names:
    fig, ax = plt.subplots(2)
    fig.suptitle(f'Plots for symbol {symbol_name}')
    mid_price_ax = sns.lineplot(data=mid_prices[symbol_name][sample_indexes],
                                ax=ax[0])
    mid_price_ax.set(xticklabels=[], xlabel="Time", ylabel="Mid price")

    size_ax = sns.lineplot(data=transacted_volumes[symbol_name][sample_indexes_transacted],
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
    nn_distances, nn_indices = NearestNeighbors(n_neighbors=num_nn + 1, metric="manhattan", algorithm="auto").fit(
        manifold).kneighbors(manifold)
    nn_distances, nn_indices = nn_distances[:, 1:], nn_indices[:, 1:]

    w = np.exp(-nn_distances) / \
        (np.sum(np.exp(-nn_distances)))

    return w, nn_indices


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
    nn_S = []
    for i, manifold in enumerate(manifolds):
        #print(s[i].shape)
        #print(nn_indices[i].shape)
        #print(s[i][nn_indices[i]].shape)
        #print(w[i].shape)
        print(w[i].shape)
        print(s[i].shape)
        #print(w[i].shape, s[i][nn_indices[i], :].shape)
        nn_S += [np.sum([w[i] * s[i][nn_indices[i]]])]

    # w = [np.exp(np.abs()) for manifold in manifolds]
    print("Ok")


PC([mid_prices[symbol_name][sample_indexes] for symbol_name in symbols_names])
