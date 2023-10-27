import os
import argparse

import nifty8 as ift
import nifty8.re as jft
import xubik0 as xu

from jax import config, random

config.update('jax_enable_x64', True)

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for eROSITA inference.",
                    nargs='?', const=1, default="eROSITA_config.yaml")
args = parser.parse_args()

if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = xu.get_config(config_path)
    file_info = cfg['files']
    ift.random.push_sseq_from_seed(cfg['seed'])

    # Sanity Checks
    if (cfg['minimization']['resume'] and cfg['mock']) and (not cfg['load_mock_data']):
        raise ValueError(
            'Resume is set to True on mock run. This is only possible if the mock data is loaded '
            'from file. Please set load_mock_data=True')

    if cfg['load_mock_data'] and not cfg['mock']:
        print('WARNING: Mockrun is set to False: Actual data is loaded')

    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        raise FileExistsError("Resume is set to False but output directory exists already!")

    # Load sky model
    sky_dict = xu.create_sky_model_from_config(config_path)
    pspec = sky_dict.pop('pspec')

    # Save config
    xu.save_config(cfg, os.path.basename(config_path), file_info['res_dir'])

    # Generate loglikelihood
    log_likelihood = xu.generate_erosita_likelihood_from_config(config_path) @ sky_dict['sky']

    # Minimization
    minimization_config = cfg['minimization']
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    pos_init = jft.Vector(jft.random_like(subkey, sky_dict['sky'].domain))

    absdelta = 1e-4 * cfg['grid']['npix']  # FIXME: Replace by domain information
    n_newton_iterations = 10
    minimization_kwargs = {"absdelta": absdelta, "maxiter": n_newton_iterations}
    linear_sampling_kwargs = {"absdelta": absdelta / 10., "maxiter": 100}
    pos, samples = jft.optimize_kl(log_likelihood,
                                   pos_init,
                                   key=key,
                                   minimization_kwargs=minimization_kwargs,
                                   sampling_cg_kwargs=linear_sampling_kwargs,
                                   **minimization_config)
    exit()
    print("Likelihood residual(s)")
    print(jft.reduced_chisq_stats(pos, samples, func=log_likelihood.normalized_residual))
    print("Prior residual(s)")
    print(jft.reduced_chisq_stats(pos, samples))
