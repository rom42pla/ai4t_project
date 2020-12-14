import os
from os.path import join, splitext
from shutil import copyfile

# path of our custom configurations
custom_configs_path = "../custom_configs"

# loops through each custom configuration
for filename in os.listdir(custom_configs_path):
    filepath = join(custom_configs_path, filename)
    if splitext(filename)[-1] == ".py":
        # copies the configuration to ABIDES' folder
        copyfile(filepath, join("..", "abides", "config", filename))
