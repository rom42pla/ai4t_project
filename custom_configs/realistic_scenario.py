from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.etf.EtfPrimaryAgent import EtfPrimaryAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.examples.ImpactAgent import ImpactAgent
from agent.etf.EtfArbAgent import EtfArbAgent
from agent.etf.EtfMarketMakerAgent import EtfMarketMakerAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from util import util

import numpy as np
import pandas as pd
import sys

DATA_DIR = "~/data"

# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for momentum config.')
# configuration settings
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('-b', '--book_freq', default=0,
                    help='Frequency at which to archive order book for visualization')
parser.add_argument('-l', '--log_dir', default="realistic_scenario",
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-sc', '--scale', type=float, default=0.5,
                    help='Scale of the simulation (1 for all number of agents)')
parser.add_argument('--hours', type=float, default=8,
                    help='hours of simulation to reproduce,'
                         'starting always from 09:00AM up to 17:00PM')
parser.add_argument('-o', '--log_orders', action='store_false', default=False,
                    help='Log every order-related action by every agent.')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='numpy.random.seed() for simulation')
# kernel and oracle settings
parser.add_argument('-r', '--shock_variance', type=float, default=500000,
                    help='Shock variance for mean reversion process (sigma^2_s)')
# agents settings
parser.add_argument('-n', '--obs_noise', type=float, default=1000000,
                    help='Observation noise variance for zero intelligence agents (sigma^2_n)')

parser.add_argument('--num_impacts', type=int, default=1,
                    help='number of impacts on the ETF, '
                         'equally distributed during the simulation')
parser.add_argument('--impacts_greed', type=float, default=0.5,
                    help='percentage of money used by impact agents')

args, remaining_args = parser.parse_known_args()

seed = args.seed
# Requested log directory.
log_dir = args.log_dir + str(seed)
# Requested order book snapshot archive frequency.
book_freq = args.book_freq
# Observation noise variance for zero intelligence agents.
sigma_n = args.obs_noise
# Shock variance of mean reversion process.
sigma_s = args.shock_variance

if seed is not None:
    seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
# Also suppresses formatting of limit orders (which is time consuming).
util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

# Config parameter that causes every order-related action to be logged by
# every agent.  Activate only when really needed as there is a significant
# time penalty to all that object serialization!
log_orders = False  # args.log_orders

print("Silent mode: {}".format(util.silent_mode))
print("Shock variance: {:0.4f}".format(sigma_s))
print("Configuration seed: {}\n".format(seed))

'''
MARKET
'''
# primary and secondary markets' hours
midnight, hours = pd.to_datetime('2020-06-15'), args.hours
assert 1 <= hours <= 8
primary_market_open, primary_market_close = midnight + pd.to_timedelta('17:00:00'), \
                                            midnight + pd.to_timedelta('17:30:00')
secondary_market_open = midnight + pd.to_timedelta('09:00:00')
secondary_market_close = secondary_market_open + pd.Timedelta(hours, unit="hours")
# symbols considered in the simulation
symbols = {'SYM1': {'r_bar': 100000, 'kappa': 1.67e-13, 'sigma_s': 0, 'type': util.SymbolType.Stock,
                    'fund_vol': 1e-4,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))},
           'SYM2': {'r_bar': 100000, 'kappa': 1.67e-13, 'sigma_s': 0, 'type': util.SymbolType.Stock,
                    'fund_vol': 1e-4,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))},
           'SYM3': {'r_bar': 100000, 'kappa': 1.67e-13, 'sigma_s': 0, 'type': util.SymbolType.Stock,
                    'fund_vol': 1e-4,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))},
           'ETF': {
               'portfolio': {'SYM1': 0.6, 'SYM2': 0.3, 'SYM3': 0.1},
               'kappa': 3 * 1.67e-13, 'sigma_s': 0,
               'fund_vol': 1e-4,
               'megashock_lambda_a': 2.77778e-13,
               'megashock_mean': 0,
               'megashock_var': 5e4,
               'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
               'type': util.SymbolType.ETF}
           }
# lists of names of stocks and ETF to distinguish between them
etfs_names, stocks_names = [symbol for symbol, infos in symbols.items() if "portfolio" in infos.keys()], \
                           [symbol for symbol, infos in symbols.items() if "portfolio" not in infos.keys()]
for etf_name in etfs_names:
    assert np.isclose(np.sum(list(symbols[etf_name]["portfolio"].values())), 1)
    symbols[etf_name]["r_bar"] = sum([share * symbols[symbol_name]["r_bar"]
                                      for symbol_name, share in symbols[etf_name]["portfolio"].items()])
symbols_full = symbols.copy()
'''
KERNEL
'''
scale = args.scale
starting_cents = 100000 * 100  # cash is always in cents
kernelStartTime, kernelStopTime = midnight, midnight + pd.to_timedelta('22:00:00')
defaultComputationDelay = 0  # no delay for this config

kernel = Kernel("Base Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
oracle = SparseMeanRevertingOracle(secondary_market_open, secondary_market_close, symbols)

'''
AGENTS
'''
# structures to hold agents' instances
agents, agent_types = [], []
# exchange agents
num_exchange_agents = 1
# ETF primary agents
num_etf_primary_agents = 1
# noise agents
num_noise_agents = 0  # int(np.ceil(scale * 5000))
noise_mkt_open, noise_mkt_close = secondary_market_open, \
                                  secondary_market_close
# value agents
num_value_agents = int(np.ceil(scale * 100))
kappa = 1.67e-15
lambda_a = 7e-11
# momentum agents
num_momentum_agents = 0  # int(np.ceil(scale * 25))
# etf arbitrage agents
num_etf_arbitrage_agents = int(np.ceil(scale * 100))
etf_arbitrage_agents_gamma = 250
# ETF market maker agents
num_etf_market_maker_agents = int(np.ceil(scale * 10))
etf_market_maker_agents_gamma = 250
# market maker agents
num_pov_market_maker_agents = int(np.ceil(scale * 1))
market_maker_agents_gamma = 100
# pov execution agents
num_pov_execution_agents = int(np.ceil(scale * 0))
pov_agent_start_time, pov_agent_end_time = secondary_market_open + pd.to_timedelta('00:30:00'), \
                                           secondary_market_close - pd.to_timedelta('00:30:00')
pov_proportion_of_volume, pov_quantity, pov_frequency, pov_direction = 0.1, 12e5, "1min", "BUY"


# impact agents
def dates_linspace(start, stop, num):
    if not isinstance(start, pd.Timestamp):
        start = midnight + pd.to_timedelta(start)
    if not isinstance(stop, pd.Timestamp):
        stop = midnight + pd.to_timedelta(stop)
    return list([str(l.time()) for l in pd.to_datetime(np.linspace(start.value,
                                                                   stop.value,
                                                                   num=num, endpoint=True))])


num_impacts, impacts_greed = args.num_impacts, \
                             args.impacts_greed
assert num_impacts >= 0
assert 0 < impacts_greed <= 1

impacts = {
    "SYM1": [{"starting_cash": starting_cents * 2,
              "time": subimpact_time,
              "symbol": "ETF",
              "greed": impacts_greed}
             for impact_time in dates_linspace(start=secondary_market_open + pd.Timedelta(0, unit="minutes"),
                                               stop=secondary_market_close,
                                               num=num_impacts + 2)[1:-1]
             for subimpact_time in dates_linspace(start=impact_time - pd.Timedelta(30, unit="seconds"),
                                                  stop=impact_time + pd.Timedelta(30, unit="seconds"),
                                                  num=6)]
}
assert set(impacts.keys()).issubset(set(symbols_full.keys()))

'''
P R I N T S
'''
print(f"Scale of simulation: {scale}")
print(pd.DataFrame(
    data={
        "event": ["Primary open", "Primary close",
                  "Secondary open", "Secondary close"],
        "date": [primary_market_open, primary_market_close,
                 secondary_market_open, secondary_market_close]
    }
).sort_values(by=['date']).to_string(index=False))
print(pd.DataFrame(
    data={
        "agent": ["Exchange",
                  "POV Market Maker", "POV Execution",
                  "Noise", "Value", "Momentum",
                  "ETF Primary", "ETF Market Maker", "ETF Arbitrage"],
        "amount": [num_exchange_agents,
                   num_pov_market_maker_agents, num_pov_execution_agents,
                   num_noise_agents, num_value_agents, num_momentum_agents,
                   num_etf_primary_agents, num_etf_market_maker_agents, num_etf_arbitrage_agents]
    }
).to_string(index=False))

df = pd.DataFrame()
for etf in impacts:
    df_etf = pd.DataFrame.from_dict(impacts[etf])
    df_etf["symbol"] = etf
    df = pd.concat([df, df_etf])
print(df.sort_values(by=["symbol", "time"]).to_string(index=False))

'''
EXCHANGE AGENTS
'''
agents.append(
    ExchangeAgent(len(agents), "Exchange Agent {}".format(len(agents)), "ExchangeAgent",
                  secondary_market_open, secondary_market_close,
                  list(symbols_full.keys()), log_orders=True, pipeline_delay=0,
                  computation_delay=0, stream_history=10, book_freq=book_freq,
                  random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))))
agent_types.append("ExchangeAgent")

for symbol_name, infos in symbols_full.items():
    # this variable tells us if the symbol is a stock or an ETF
    is_ETF = symbol_name in etfs_names
    # computes some parameters for each symbol
    r_bar = infos["r_bar"] if not is_ETF \
        else np.sum([symbols_full[symbol]["r_bar"] * share for symbol, share in infos["portfolio"].items()])

    '''
    POV EXECUTION AGENTS
    '''
    for i in range(num_pov_execution_agents):
        agents.append(POVExecutionAgent(id=len(agents),
                                        name='POVExecutionAgent {}'.format(len(agents)),
                                        type='ExecutionAgent',
                                        symbol=symbol_name, starting_cash=starting_cents,
                                        start_time=pov_agent_start_time, end_time=pov_agent_end_time,
                                        freq=pov_frequency,
                                        lookback_period=pov_frequency,
                                        pov=pov_proportion_of_volume,
                                        direction=pov_direction,
                                        quantity=pov_quantity,
                                        log_orders=log_orders,  # needed for plots so conflicts with others
                                        random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64'))))
        agent_types.append("ExecutionAgent")

    '''
    IMPACT AGENTS
    '''
    if symbol_name in impacts.keys():
        for impact_dict in impacts[symbol_name]:
            agents.append(
                ImpactAgent(len(agents), "Impact Agent {} {}".format(symbol_name, len(agents)),
                            "ImpactAgent{}{}".format(symbol_name, len(agents)),
                            symbol=symbol_name, starting_cash=impact_dict["starting_cash"], greed=impact_dict["greed"],
                            impact=True, impact_time=midnight + pd.to_timedelta(impact_dict["time"]),
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("ImpactAgent {}".format(len(agents)))

    '''
    NOISE AGENTS
    '''
    for i in range(num_noise_agents if not is_ETF else int(num_noise_agents * 1.5)):
        agents.append(NoiseAgent(id=len(agents), name="NoiseAgent {}".format(len(agents)), type="NoiseAgent",
                                 symbol=symbol_name, starting_cash=starting_cents,
                                 wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(
                                     seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))))
        agent_types.append('NoiseAgent')

    '''
    VALUE AGENTS
    '''
    for i in range(num_value_agents):
        agents.append(ValueAgent(id=len(agents), name="Value Agent {}".format(len(agents)), type="ValueAgent",
                                 symbol=symbol_name, starting_cash=starting_cents,
                                 sigma_n=sigma_n,
                                 r_bar=r_bar,
                                 kappa=kappa,
                                 lambda_a=lambda_a,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(
                                     seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))))
        agent_types.append('ValueAgent')

    '''
    MOMENTUM AGENTS
    '''
    for i in range(num_momentum_agents):
        agents.append(
            MomentumAgent(len(agents), "Momentum Agent {}".format(len(agents)), type="MomentumAgent",
                          min_size=1, max_size=10, wake_up_freq='20s',
                          symbol=symbol_name, starting_cash=starting_cents,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
                          log_orders=log_orders))
        agent_types.append("MomentumAgent {}".format(len(agents)))

    '''
    POV MARKET MAKER AGENTS
    '''
    for i in range(num_pov_market_maker_agents):
        agents.append(AdaptiveMarketMakerAgent(id=len(agents),
                                               name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(len(agents)),
                                               type='AdaptivePOVMarketMakerAgent',
                                               symbol=symbol_name,
                                               starting_cash=starting_cents,
                                               pov=0.025,
                                               min_order_size=1,
                                               window_size='adaptive',
                                               num_ticks=10,
                                               wake_up_freq="10S",
                                               cancel_limit_delay=50,
                                               skew_beta=0,
                                               level_spacing=5,
                                               spread_alpha=0.75,
                                               backstop_quantity=50000,
                                               lambda_a=lambda_a,
                                               log_orders=log_orders,
                                               random_state=np.random.RandomState(
                                                   seed=np.random.randint(low=0, high=2 ** 32,
                                                                          dtype='uint64'))))
        agent_types.append('POVMarketMakerAgent')

    if is_ETF:
        portfolio = symbols_full[symbol_name]["portfolio"]

        '''
        ETF PRIMARY AGENTS
        '''
        for i in range(num_etf_primary_agents):
            agents.append(EtfPrimaryAgent(len(agents), "ETF Primary Agent {}".format(len(agents)), "EtfPrimaryAgent",
                                          primary_market_open, primary_market_close, symbol_name,
                                          pipeline_delay=0, computation_delay=0,
                                          random_state=np.random.RandomState(
                                              seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("EtfPrimeAgent")

        '''
        ETF ARBITRAGE AGENTS
        '''
        for j in range(num_etf_arbitrage_agents):
            agents.append(
                EtfArbAgent(len(agents),
                            "Etf Arb Agent {}".format(len(agents)),
                            "EtfArbAgent",
                            portfolio=portfolio, gamma=etf_arbitrage_agents_gamma,
                            starting_cash=starting_cents, lambda_a=lambda_a, log_orders=log_orders,
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("EtfArbAgent {}".format(len(agents)))

        '''
        ETF MARKET MAKER AGENTS
        '''
        for i in range(num_etf_market_maker_agents):
            strat_name = "Type {} [gamma = {}]".format(len(agents), etf_market_maker_agents_gamma)
            agents.append(EtfMarketMakerAgent(len(agents),
                                              "Etf MM Agent {} {}".format(len(agents), strat_name),
                                              "EtfMarketMakerAgent {}".format(strat_name),
                                              portfolio=portfolio, starting_cash=starting_cents,
                                              gamma=etf_market_maker_agents_gamma, lambda_a=lambda_a,
                                              log_orders=log_orders,
                                              random_state=np.random.RandomState(
                                                  seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("EtfMarketMakerAgent {}".format(i))

# This configures all agents to a starting latency as described above.
# latency = np.random.uniform(low = 21000, high = 13000000, size=(len(agent_types),len(agent_types)))
latency = np.random.uniform(low=10, high=100, size=(len(agent_types), len(agent_types)))

# Overriding the latency for certain agent pairs happens below, as does forcing mirroring
# of the matrix to be symmetric.
for i, t1 in zip(range(latency.shape[0]), agent_types):
    for j, t2 in zip(range(latency.shape[1]), agent_types):
        # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
        if j > i:
            # Arb agents should be the fastest in the market.
            if (("ExchangeAgent" in t1 and "EtfArbAgent" in t2)
                    or ("ExchangeAgent" in t2 and "EtfArbAgent" in t1)):
                # latency[i,j] = 20000
                latency[i, j] = 5
            elif (("ExchangeAgent" in t1 and "EtfMarketMakerAgent" in t2)
                  or ("ExchangeAgent" in t2 and "EtfMarketMakerAgent" in t1)):
                # latency[i,j] = 20000
                latency[i, j] = 1
            elif (("ExchangeAgent" in t1 and "ImpactAgent" in t2)
                  or ("ExchangeAgent" in t2 and "ImpactAgent" in t1)):
                # latency[i,j] = 20000
                latency[i, j] = 1

        elif i > j:
            # This "bottom" half of the matrix simply mirrors the top.
            if (("ExchangeAgent" in t1 and "EtfArbAgent" in t2)
                    or ("ExchangeAgent" in t2 and "EtfArbAgent" in t1)):
                # latency[i,j] = 20000
                latency[i, j] = 5
            elif (("ExchangeAgent" in t1 and "EtfMarketMakerAgent" in t2)
                  or ("ExchangeAgent" in t2 and "EtfMarketMakerAgent" in t1)):
                # latency[i,j] = 20000
                latency[i, j] = 1
            elif (("ExchangeAgent" in t1 and "ImpactAgent" in t2)
                  or ("ExchangeAgent" in t2 and "ImpactAgent" in t1)):
                # latency[i,j] = 20000
                latency[i, j] = 1
            else:
                latency[i, j] = latency[j, i]
        else:
            # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
            # takes about 20 microseconds.
            # latency[i,j] = 10000
            latency[i, j] = 1

# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
# In this config, there is no latency (noisy or otherwise).
noise = [0.0]

# Start the kernel running.
kernel.runner(agents=agents, startTime=kernelStartTime,
              stopTime=kernelStopTime, agentLatency=latency,
              latencyNoise=noise,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle, log_dir=log_dir)
