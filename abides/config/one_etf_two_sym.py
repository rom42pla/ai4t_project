from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.etf.EtfPrimaryAgent import EtfPrimaryAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.examples.ImpactAgent import ImpactAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.etf.EtfArbAgent import EtfArbAgent
from agent.etf.EtfMarketMakerAgent import EtfMarketMakerAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from util.order import LimitOrder
from util.oracle.MeanRevertingOracle import MeanRevertingOracle
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
parser.add_argument('-b', '--book_freq', default=0,
                    help='Frequency at which to archive order book for visualization')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-g', '--greed', type=float, default=0.25,
                    help='Impact agent greed')
parser.add_argument('-i', '--impact', action='store_false',
                    help='Do not actually fire an impact trade.', default=True)
parser.add_argument('-l', '--log_dir', default="twosym",
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n', '--obs_noise', type=float, default=1000000,
                    help='Observation noise variance for zero intelligence agents (sigma^2_n)')
parser.add_argument('-r', '--shock_variance', type=float, default=500000,
                    help='Shock variance for mean reversion process (sigma^2_s)')
parser.add_argument('-o', '--log_orders', action='store_true', default=True,
                    help='Log every order-related action by every agent.')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

seed = args.seed
# Requested log directory.
log_dir = args.log_dir + str(seed)
# Requested order book snapshot archive frequency.
book_freq = args.book_freq
# Observation noise variance for zero intelligence agents.
sigma_n = args.obs_noise
# Shock variance of mean reversion process.
sigma_s = args.shock_variance
# Impact agent greed.
greed = args.greed
# Should the impact agent actually trade?
impact = args.impact

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
log_orders = args.log_orders

print("Silent mode: {}".format(util.silent_mode))
print("Logging orders: {}".format(log_orders))
print("Book freq: {}".format(book_freq))
print("ZeroIntelligenceAgent noise: {:0.4f}".format(sigma_n))
print("ImpactAgent greed: {:0.2f}".format(greed))
print("ImpactAgent firing: {}".format(impact))
print("Shock variance: {:0.4f}".format(sigma_s))
print("Configuration seed: {}\n".format(seed))

'''
MARKET
'''
# primary and secondary markets' hours
midnight = pd.to_datetime('2020-06-15')
primary_market_open, primary_market_close = midnight + pd.to_timedelta('04:00:00'), \
                                            midnight + pd.to_timedelta('09:29:59')
secondary_market_open, secondary_market_close = midnight + pd.to_timedelta('09:30:00'), \
                                                midnight + pd.to_timedelta('11:00:00')
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
           'ETF': {'r_bar': 100000, 'kappa': 2 * 1.67e-13, 'sigma_s': 0, 'portfolio': ['SYM1', 'SYM2', 'SYM3'],
                   'fund_vol': 1e-4,
                   'megashock_lambda_a': 2.77778e-13,
                   'megashock_mean': 0,
                   'megashock_var': 5e4,
                   'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                   'type': util.SymbolType.ETF}
           }
symbols_full = symbols.copy()
# lists of names of stocks and ETF to distinguish between them
etfs_names, stocks_names = [symbol for symbol, infos in symbols_full.items() if "portfolio" in infos.keys()], \
                           [symbol for symbol, infos in symbols_full.items() if "portfolio" not in infos.keys()]
'''
KERNEL
'''
starting_cents = 10000000
kernelStartTime, kernelStopTime = midnight, midnight + pd.to_timedelta('20:00:00')
defaultComputationDelay = 0  # no delay for this config

kernel = Kernel("Base Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
oracle = SparseMeanRevertingOracle(secondary_market_open, secondary_market_close, symbols)

'''
AGENTS
'''
# structures to hold agents' instances
num_agents, agents, agent_types = 0, [], []
# exchange agents
num_exchange_agents = 1
# ETF primary agents
num_etf_primary_agents = 1
# noise agents
num_noise_agents = 10
noise_mkt_open, noise_mkt_close = secondary_market_open + pd.to_timedelta("00:15:00"), \
                                  secondary_market_close - pd.to_timedelta("00:15:00")
# value agents
num_value_agents = 10
kappa = 1.67e-15
lambda_a = 7e-11
# momentum agents
num_momentum_agents = 3
momentum_lookback = 10
# etf arbitrage agents
num_etf_arbitrage_agents = 10
etf_arbitrage_agents_gamma = 250
# market maker agents
num_etf_market_maker_agents, num_pov_market_maker_agents = 1, 0
etf_market_maker_agents_gamma = 250
market_maker_agents_gamma = 100
# pov execution agents
num_pov_execution_agents = 0
pov_agent_start_time, pov_agent_end_time = secondary_market_open + pd.to_timedelta('00:30:00'), \
                                           secondary_market_close - pd.to_timedelta('00:30:00')
pov_proportion_of_volume, pov_quantity, pov_frequency, pov_direction = 0.1, 12e5, "1min", "BUY"
# impact agents
impacts = {}
assert set(impacts.keys()).issubset(set(symbols_full.keys()))

'''
EXCHANGE AGENTS
'''
agents.append(
    ExchangeAgent(num_agents, "Exchange Agent {}".format(num_agents), "ExchangeAgent",
                  secondary_market_open, secondary_market_close,
                  list(symbols_full.keys()), log_orders=log_orders, pipeline_delay=0,
                  computation_delay=0, stream_history=10, book_freq=book_freq,
                  random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))))
agent_types.append("ExchangeAgent")
num_agents += 1

for symbol_name, infos in symbols_full.items():
    # this variable tells us if the symbol is a stock or an ETF
    is_ETF = symbol_name in etfs_names
    # computes some parameters for each symbol
    r_bar = infos["r_bar"] if not is_ETF \
        else np.sum([symbols_full[symbol]["r_bar"] for symbol in infos["portfolio"]])

    '''
    POV EXECUTION AGENTS
    
    for i in range(num_pov_execution_agents):
        agents.append(POVExecutionAgent(id=num_agents,
                                        name='POVExecutionAgent {}'.format(num_agents),
                                        type='ExecutionAgent',
                                        symbol=symbol_name, starting_cash=starting_cents,
                                        start_time=pov_agent_start_time, end_time=pov_agent_end_time,
                                        freq=pov_frequency,
                                        lookback_period=pov_frequency,
                                        pov=pov_proportion_of_volume,
                                        direction=pov_direction,
                                        quantity=pov_quantity,
                                        log_orders=True,  # needed for plots so conflicts with others
                                        random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64'))))
        agent_types.append("ExecutionAgent")
        num_agents += 1 '''

    '''
    IMPACT AGENTS
    
    if symbol_name in impacts.keys():
        for itrades in impacts[symbol_name]:
            agents.append(
                ImpactAgent(num_agents, "Impact Agent {} {}".format(symbol_name, num_agents),
                            "ImpactAgent{}{}".format(symbol_name, num_agents),
                            symbol=symbol_name, starting_cash=starting_cents, greed=greed,
                            impact=impact, impact_time=midnight + pd.to_timedelta(itrades),
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("ImpactAgent {}".format(num_agents))
            num_agents += 1'''

    '''
    NOISE AGENTS
    '''
    for i in range(num_noise_agents):
        agents.append(NoiseAgent(id=num_agents, name="NoiseAgent {}".format(num_agents), type="NoiseAgent",
                                 symbol=symbol_name, starting_cash=starting_cents,
                                 wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(
                                     seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))))
        agent_types.append('NoiseAgent')
        num_agents += 1

    '''
    VALUE AGENTS
    '''
    for i in range(num_value_agents):
        agents.append(ValueAgent(id=num_agents, name="Value Agent {}".format(num_agents), type="ValueAgent",
                                 symbol=symbol_name, starting_cash=starting_cents,
                                 sigma_n=sigma_n,
                                 r_bar=r_bar,
                                 kappa=kappa,
                                 lambda_a=lambda_a,
                                 log_orders=log_orders,
                                 random_state=np.random.RandomState(
                                     seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))))
        agent_types.append('ValueAgent')
        num_agents += 1

    '''
    MOMENTUM AGENTS
    '''
    for i in range(num_momentum_agents):
        agents.append(
            MomentumAgent(num_agents, "Momentum Agent {}".format(num_agents), type="MomentumAgent",
                          min_size=1, max_size=10, wake_up_freq='20s',
                          symbol=symbol_name, starting_cash=starting_cents,
                          # lookback=lookback, -> al limite inserire in Trading Agent, per ora tolto
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
                          log_orders=log_orders))
        agent_types.append("MomentumAgent {}".format(num_agents))
        num_agents += 1

    if not is_ETF:
        '''
        POV MARKET MAKER AGENTS
        '''
        for i in range(num_pov_market_maker_agents):
            agents.append(AdaptiveMarketMakerAgent(id=num_agents,
                                                   name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(num_agents),
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
                                                   log_orders=log_orders,
                                                   random_state=np.random.RandomState(
                                                       seed=np.random.randint(low=0, high=2 ** 32,
                                                                              dtype='uint64'))))
            agent_types.append('POVMarketMakerAgent')
            num_agents += 1

    elif is_ETF:
        portfolio = symbols_full[symbol_name]["portfolio"]

        '''
        ETF PRIMARY AGENTS
        '''
        for i in range(num_etf_primary_agents):
            agents.append(EtfPrimaryAgent(num_agents, "ETF Primary Agent {}".format(num_agents), "EtfPrimaryAgent",
                                          primary_market_open, primary_market_close, 'ETF',
                                          pipeline_delay=0, computation_delay=0,
                                          random_state=np.random.RandomState(
                                              seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("EtfPrimeAgent")
            num_agents += 1

        '''
        ETF ARBITRAGE AGENTS
        '''
        for j in range(num_etf_arbitrage_agents):
            agents.append(
                EtfArbAgent(num_agents,
                            "Etf Arb Agent {}".format(num_agents),
                            "EtfArbAgent",
                            portfolio=portfolio, gamma=etf_arbitrage_agents_gamma,
                            starting_cash=starting_cents, lambda_a=1e-9, log_orders=log_orders,
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("EtfArbAgent {}".format(num_agents))
            num_agents += 1

        '''
        ETF MARKET MAKER AGENTS
        '''
        for i in range(num_etf_market_maker_agents):
            strat_name = "Type {} [gamma = {}]".format(num_agents, etf_market_maker_agents_gamma)
            agents.append(EtfMarketMakerAgent(num_agents,
                                              "Etf MM Agent {} {}".format(num_agents, strat_name),
                                              "EtfMarketMakerAgent {}".format(strat_name),
                                              portfolio=portfolio, starting_cash=starting_cents,
                                              gamma=etf_market_maker_agents_gamma, lambda_a=1e-9,
                                              log_orders=log_orders,
                                              random_state=np.random.RandomState(
                                                  seed=np.random.randint(low=0, high=2 ** 32))))
            agent_types.append("EtfMarketMakerAgent {}".format(i))
            num_agents += 1

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
