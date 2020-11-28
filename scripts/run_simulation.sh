#!/bin/bash

# updates our custom configurations into ABIDES' folder
bash update_custom_configs.sh

cd ../abides
python3 abides.py -c better_twoSymbols -s 1234

# gets the csvs from the simulation
cd ../scripts
#python get_csv_simulations.py
