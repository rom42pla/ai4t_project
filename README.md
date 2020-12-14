# AI4T 2020/21 project

## Who are we?
| Name | Matricola |
| --- | --- |
| Romeo Lanzino | 1753403 |
| Dario Ruggeri | 1741637 |

## Install the dependencies
From inside the root folder (the one containing `README.md`) run:
```bash
pip install -r abides/requirements.txt
```

## Run a simulation
You can either run a simulation or multiple ones in parallel by using `scripts/run_simulation.py` or `scripts/run_parallel_simulations.py`

### Run a single simulation
From inside the root folder, `cd` to script and run `run_simulation.py`:
```bash
cd scripts
python run_simulation.py
cd ..
```

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

### Run multiple simulations
From inside the root folder, `cd` to script and run `run_parallel_simulations.py`:
```bash
cd scripts
python run_parallel_simulations.py
cd ..
```

```
usage: run_parallel_simulations.py [-h] [--configuration CONFIGURATION]
                                   [--scale SCALE]
                                   [--num_simulations NUM_SIMULATIONS]

Detailed options for momentum config.

optional arguments:
  -h, --help            show this help message and exit
  --configuration CONFIGURATION
                        name of the configurationinside custom_configs/
  --scale SCALE         scale of the simulation
  --num_simulations NUM_SIMULATIONS
                        number of parallel simulations to generate
```
