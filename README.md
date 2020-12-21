# AI4T 2020/21 project about Shock Propagation

## Who are we?
| Name | Matricola |
| --- | --- |
| Romeo Lanzino | 1753403 |
| Dario Ruggeri | 1741637 |

## About this project
This project is indeed an analysis about the phenomenon of **shock propagation** from a symbol to another under different and controlled circumstances.

This repo allow you to **simulate a trading day** with an exchange tracking an ETF and many underlying symbols with different shares (weigths) in it.
This environment is of course populated with different agents behaving in different ways according to their strategy: for example, `ValueAgent`s try to follow the fundamental given them by an `Oracle`, `MomentumAgent`s try to follow the trends, while `NoiseAgent`s simply do controlled random orders in order to add a realistic scent to the simulation.

At the end of each simulation you'll be prompted with some **plots** of the mid price, spread and transaction volume of each symbol involved alongside the **strength of the causalities** of each symbol to the other to see if they have a positive, negative or dark correlation.


## Check the results
To check our result you'll have to install the requirements and run a simulation, at the end of which you'll be prompted with plots and causality stats for each symbol, also saved into `data\<simulation_name>`.

### Install the dependencies
You must have installed:
- Python 3.6 or 3.7 installed (3.8+ gave us some problem concerning `matplotlib`'s package') 
- `fim` image viewer if you want to have the plots prompted at the end of each simulation, else you'll have to check them manually inside `data\<simulation_name>\plots`

Assuming you've successfully installed a supported version of Python, from inside the root folder (the one containing `README.md`) type:

```bash
sudo apt install fim
pip install -r requirements.txt
```

This will install all the dependencies needed for the program to work correctly.

Please note that, currently, this project has been made and tested to run correctly in Linux environments (Linux Mint 20 and Ubuntu 20.04) since **ABIDES do seem not to work on Windows**. 

## Run a simulation
You can either run a simulation or multiple ones in sequence by using `scripts/run_simulation.py` with different parameters.

### Run a single simulation
From inside the root folder type:

```bash
cd scripts
python run_simulation.py
cd ..
```

This will start a new simulation (with a random seed) of a trading day with a secondary market duration of two hours, a huge shock in the middle, an ETF and three underlying symbols with different shares.

You can view the list of parameters by giving the `-h` parameter to the script, e.g. `python run_simulation.py -h`. Some interesting parameters:

- `--seed`: specify a seed in order to be able to reproduce a particular simulation (default is a random number)
- `--hours`: number of hours of market, from 1 to 8 (default is 2)
- `--scale`: multiplier of the number of agents, useful to reduce impact on the RAM (default is 0.3 the full number of agents)
- `--num_impacts`: number of shocks on the non-ETF symbol with biggest share in it, evenly distributed in time (default is 1)




