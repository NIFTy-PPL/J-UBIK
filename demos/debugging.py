# %%
import os
import argparse
from os.path import join

import nifty8.re as jft
import jubik0 as ju

from jax import config, random, jit

config.update('jax_enable_x64', True)

# Parser Setup
#parser = argparse.ArgumentParser()
#parser.add_argument('config', type=str, help="Config file (.yaml) for eROSITA inference.",
#                    nargs='?', const=1, default="eROSITA_config.yaml")
#args = parser.parse_args()
# %%

# %%
# Load config file
config_path = "eROSITA_config.yaml"
cfg = ju.get_config(config_path)
file_info = cfg['files']

if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
    file_info["res_dir"] = file_info["res_dir"] + "_new"
    print("FYI: Resume is set to False, but the output directory already exists. "
            "The result_dir has been appended with the string *new*.")
    
#%%

# Save run configuration
ju.save_config_copy(os.path.basename(config_path), output_dir=file_info['res_dir'])
# ju.save_local_packages_hashes_to_txt(['jubik0', 'nifty8'], # FIXME: fix for cluster
#                                      join(file_info['res_dir'], "packages_hashes.txt"),
#                                      paths_to_git=[os.path.dirname(os.getcwd()), None],
#                                      verbose=False)

# Load sky model
sky_model = ju.SkyModel(config_path)
sky = sky_model.create_sky_model()
sky_dict = sky_model.sky_model_to_dict()
# %%
# Generate eROSITA data (if it does not alread exist)
ju.create_erosita_data_from_config(config_path)
# %%
# Minimization
minimization_config = cfg['minimization']
key = random.PRNGKey(cfg['seed'])
key, subkey = random.split(key)
pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))
# %%
msky = sky(pos_init)
# %%
R_dict = ju.build_erosita_response_from_config(config_path)

# %%
##### NOW Open Htop

# %%
jpsf = jit(R_dict["psf"])
jpsf(msky)
# %%
# Generate loglikelihood (Building masked (mock) data and response)
log_likelihood = ju.generate_erosita_likelihood_from_config(config_path).amend(sky)


# %%
