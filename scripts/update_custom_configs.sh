#!/bin/bash

# updates abides/config with our custom configs
for f in ../custom_configs/*.py
do
  cp $f ../abides/config/${f##*/}
done
