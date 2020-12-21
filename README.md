# AI4T 2020/21 project

## Who are we?
| Name | Matricola |
| --- | --- |
| Romeo Lanzino | 1753403 |
| Dario Ruggeri | 1741637 |

## Install the dependencies
From inside the root folder (the one containing `README.md`) run:
```bash
pip install -r requirements.txt
```
This will install all the dependencies needed for the program to work correctly.

Please note that, currently, this project has been made in Linux environments (Linux Mint 20 and Ubuntu 20.04) since **ABIDES seems not to work on Windows**. 

## Run a simulation
You can either run a simulation or multiple ones in sequence by using `scripts/run_simulation.py`

### Run a single simulation
From inside the root folder:

```bash
cd scripts
python run_simulation.py
cd ..
```

This will start a new simulation (random seed) of a trading day with a secondary market duration of two hours, a huge shock in the middle, an ETF and three underlying symbols with different shares.



```
usage: run_simulation.py [-h] [--configuration CONFIGURATION] [--seed SEED]
                         [--scale SCALE] [--plot_config PLOT_CONFIG]

Run an ABIDES simulation

optional arguments:
  -h, --help            show this help message and exit
  --configuration CONFIGURATION
                        name of the configuration to use,inside
                        custom_configs/
  --seed SEED           seed of the simulation
  --scale SCALE         scale of the simulation (proportion of agents wrt full
                        simulation)
  --plot_config PLOT_CONFIG
                        name of the .json with the parameters of the
                        plot,inside abides/util/plotting/configs/
```
