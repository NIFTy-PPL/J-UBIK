import os
import argparse

import nifty8.re as jft
import jubik0 as ju

from jax import config, random

config.update('jax_enable_x64', True)

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for JWST inference.",
                    nargs='?', const=1, default="JWST_config.yaml")
args = parser.parse_args()

if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

    # Sanity Checks
    if ((cfg['minimization']['resume'] and cfg.get('mock', False)) and
            (not cfg.get('load_mock_data', False))):
        raise ValueError(
            'Resume is set to True on mock run. This is only possible if the mock data is loaded '
            'from file. Please set load_mock_data=True')

    if cfg.get('load_mock_data', False) and not cfg.get('mock', False):
        print('WARNING: Mockrun is set to False: Actual data is loaded')

    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        file_info["res_dir"] = file_info["res_dir"] + "_new"
        print("FYI: Resume is set to False, but the output directory already exists. "
              "The result_dir has been appended with the string *new*.")

    # Load sky model
    sky_dict = ju.create_sky_model_from_config(config_path)

    # Save config
    ju.save_config(cfg, os.path.basename(config_path), file_info['res_dir'])

    # Generate loglikelihood
    log_likelihood, data = ju.generate_jwst_likelihood_from_config(
        sky_dict, config_path)

    pspec = sky_dict.pop('pspec')
    _ = sky_dict.pop('target')

    # Minimization
    minimization_config = cfg['minimization']
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    pos_init = 0.1 * \
        jft.Vector(jft.random_like(subkey, log_likelihood.domain))

    kl_solver_kwargs = minimization_config.pop('kl_kwargs')
    # FIXME: Replace by domain information
    kl_solver_kwargs['minimize_kwargs']['absdelta'] *= cfg['grid']['npix']

    s = sky_dict['sky'](pos_init)

    # Plot
    def plot(s, x): return ju.plot_sample_and_stats(file_info["res_dir"],
                                                    sky_dict,
                                                    s,
                                                    iteration=x.nit)

    samples, state = jft.optimize_kl(log_likelihood,
                                     pos_init,
                                     key=key,
                                     kl_kwargs=kl_solver_kwargs,
                                     callback=plot,
                                     odir=file_info["res_dir"],
                                     **minimization_config
                                     )
