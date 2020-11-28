#!/bin/bash

sim_name="one_etf_two_sym"
config_name=$sim_name
seed_num=1234
sim_path="../../../data/"$sim_name$seed_num"/"

img_path=$sim_path"plots/"
plot_config="10min_plot_config"

# updates our custom configurations into ABIDES' folder
bash update_custom_configs.sh

# running the simulation:
cd ../abides
 python -u abides.py -c $config_name -l $sim_name -b 0 -s $seed_num -o True

# plotting the results: (one image for each symbol)
cd util/plotting/
mkdir -p $img_path
python -u liquidity_telemetry.py ../../log/$sim_name$seed_num/ExchangeAgent0.bz2 ../../log/$sim_name$seed_num/ORDERBOOK_ETF_FULL.bz2 -o $img_path"ETF_plot.png" -c configs/$plot_config".json"
python -u liquidity_telemetry.py ../../log/$sim_name$seed_num/ExchangeAgent0.bz2 ../../log/$sim_name$seed_num/ORDERBOOK_SYM1_FULL.bz2 -o $img_path"SYM1_plot.png" -c configs/$plot_config".json"
python -u liquidity_telemetry.py ../../log/$sim_name$seed_num/ExchangeAgent0.bz2 ../../log/$sim_name$seed_num/ORDERBOOK_SYM2_FULL.bz2 -o $img_path"SYM2_plot.png" -c configs/$plot_config".json"
cd ../..



# gets the csvs from the simulation
cd ../scripts
#python get_csv_simulations.py
