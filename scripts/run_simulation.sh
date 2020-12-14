#!/bin/bash

sim_name="realistic_scenario"
scale="0.5"

while getopts c:s: flag; do
  case "${flag}" in
  s) scale=${OPTARG} ;;
  c) sim_name=${OPTARG} ;;
  *) sim_name="realistic_scenario" ;;
  esac
done

clear

echo "Running simulation $sim_name"

config_name=$sim_name
seed_num=1234
sim_path="../../../data/"$sim_name$seed_num"/"

img_path=$sim_path"plots/"
plot_config="plot_09.30_11.30"

# updates our custom configurations into ABIDES' folder
bash update_custom_configs.sh

# running the simulation:
cd ../abides
python -u abides.py -c $config_name -l $sim_name -b 0 -s $seed_num -o True -sc $scale

# plotting the results: (one image for each symbol)
cd util/plotting/
mkdir -p $img_path
python -u liquidity_telemetry.py ../../log/$sim_name$seed_num/ExchangeAgent0.bz2 ../../log/$sim_name$seed_num/ORDERBOOK_ETF_FULL.bz2 -o $img_path"ETF_plot.png" -c configs/$plot_config".json"
python -u liquidity_telemetry.py ../../log/$sim_name$seed_num/ExchangeAgent0.bz2 ../../log/$sim_name$seed_num/ORDERBOOK_SYM1_FULL.bz2 -o $img_path"SYM1_plot.png" -c configs/$plot_config".json"
python -u liquidity_telemetry.py ../../log/$sim_name$seed_num/ExchangeAgent0.bz2 ../../log/$sim_name$seed_num/ORDERBOOK_SYM2_FULL.bz2 -o $img_path"SYM2_plot.png" -c configs/$plot_config".json"
python -u liquidity_telemetry.py ../../log/$sim_name$seed_num/ExchangeAgent0.bz2 ../../log/$sim_name$seed_num/ORDERBOOK_SYM3_FULL.bz2 -o $img_path"SYM3_plot.png" -c configs/$plot_config".json"
cd ../..

# shows the plots
cd ..
fim data/$sim_name$seed_num/plots/

# gets the .csv from the simulation
#python get_csv_simulations.py
