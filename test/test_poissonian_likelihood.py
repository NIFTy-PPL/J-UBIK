import os
import argparse
import dataclasses
from typing import Any
import nifty8.re as jft
import nifty8 as ift
import jubik0 as ju
import xubik0 as xu

from jax import config, random

config.update('jax_enable_x64', True)

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for eROSITA inference.",
                    nargs='?', const=1, default="eROSITA_config.yaml")
args = parser.parse_args()


class FullModel(jft.Model):
    kern: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, kern, instrument, pre_ops):
        self.instrument = instrument
        self.kern = kern
        self.pre_ops = pre_ops
        super().__init__(init=self.pre_ops.init)

    def __call__(self, x):
        return self.instrument(x=self.pre_ops(x), k=self.kern)


def generate_erosita_likelihood_from_config(sky, response_dict, masked_data):
    # Load data files
    response_func = response_dict['R']
    full_model = FullModel(kern=response_dict["kernel_arr"],
                           instrument=response_func,
                           pre_ops=sky)
    return jft.Poissonian(masked_data).amend(full_model)

def test_likelihood_sum(config_paths, sky, masked_full_data):
    lklhd = None
    for key, item in masked_full_data.tree.items():
        response_dict = ju.build_erosita_response_from_config(config_paths[key])
        masked_data = jft.Vector({key:item})
        response_func = response_dict['R']
        full_model = FullModel(kern=response_dict["kernel_arr"],
                               instrument=response_func,
                               pre_ops=sky)

        if lklhd is None:
            lklhd = jft.Poissonian(masked_data).amend(full_model)
        else:
            lklhd = lklhd + jft.Poissonian(masked_data).amend(full_model)
    return lklhd


if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        file_info["res_dir"] = file_info["res_dir"] + "_new"
        print("FYI: Resume is set to False, but the output directory already exists. "
              "The result_dir has been appended with the string *new*.")

    # Save run configuration
    ju.save_config_copy(os.path.basename(config_path), output_dir=file_info['res_dir'])
    # Load sky model
    sky_model = ju.SkyModel(config_path)
    old_sky = sky_model.create_sky_model()
    sky = jft.Model(lambda x: old_sky(x), domain=jft.Vector(old_sky.domain))
    sky_dict = sky_model.sky_model_to_dict()

    # Generate loglikelihood (Building masked (mock) data and response)
    response_dict = ju.build_erosita_response_from_config(config_path)
    ju.create_data_from_config(config_path, response_dict)
    masked_data = ju.load_masked_data_from_config(config_path)
    config_paths = {1: 'config_lklhd_test1.yaml', 2: 'config_lklhd_test2.yaml',
                    3: 'config_lklhd_test3.yaml', 4: 'config_lklhd_test4.yaml',
                    6: 'config_lklhd_test6.yaml'}
    lklhd = test_likelihood_sum(config_paths, sky, masked_data)
    log_likelihood = generate_erosita_likelihood_from_config(sky,response_dict, masked_data)

    # Set initial position
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    n = 5
    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))
    print(f'{n}tms:{log_likelihood(pos_init)}')
    print(f'{n}tms summed:{lklhd(pos_init)}')

    import numpy as np

    print(np.array_equal(lklhd.normalized_residual(pos_init), log_likelihood.normalized_residual(pos_init)))
    print(np.array_equal(lklhd(pos_init), log_likelihood(pos_init)))
    print(np.array_equal(lklhd.transformation(pos_init), log_likelihood.transformation(pos_init)))

    response_func = response_dict['R']
    full_model = FullModel(kern=response_dict["kernel_arr"],
                           instrument=response_func,
                           pre_ops=sky)


    def compare_dicts_of_arrays(dict1, dict2):
        # Check if both dictionaries have the same keys
        if set(dict1.keys()) != set(dict2.keys()):
            return False

        # Check each array for equality
        for key in dict1.keys():
            if not np.array_equal(dict1[key], dict2[key]):
                return False

        return True


    print(compare_dicts_of_arrays(log_likelihood.metric(pos_init, pos_init).tree.tree, lklhd.metric(pos_init, pos_init).tree.tree))
    print(compare_dicts_of_arrays(log_likelihood.left_sqrt_metric(pos_init, pos_init).tree.tree,
                                lklhd.left_sqrt_metric(pos_init, pos_init).tree.tree))
    print(compare_dicts_of_arrays(log_likelihood.right_sqrt_metric(full_model(pos_init), full_model(pos_init)).tree.tree,
                                lklhd.right_sqrt_metric(pos_init, pos_init).tree.tree))


