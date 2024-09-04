import os
import argparse

import nifty8.re as jft
import jubik0 as ju

from jax import config, random

config.update('jax_enable_x64', True)

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for Chandra inference.",
                    nargs='?', const=1, default="chandra_config.yaml")
args = parser.parse_args()

if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        file_info["res_dir"] = file_info["res_dir"] + "_new"
        print("FYI: Resume is set to False, but the output directory already exists. "
              "The result_dir has been appended with the string *new*.")