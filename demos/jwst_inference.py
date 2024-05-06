import os
import argparse

import nifty8.re as jft
import jubik0 as ju

from jax import config, random

from sys import exit

config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for JWST inference.",
                    nargs='?', const=1, default="JWST_config.yaml")
args = parser.parse_args()


def build_residuals(key, val):
    def response(x):
        return val.response({
            'sky': sky_dict['sky'](x),
            '_'.join((key, 'zero_flux_mean')): x['_'.join((key, 'zero_flux_mean'))]
        })
    d = val.data_2d
    n = val.noise_2d
    return lambda x: (d - response(x))/n


def build_plot_simple_residuals(data, log_data=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if log_data:
        norm = LogNorm()
    else:
        norm = Normalize()

    def plot_simple_residuals(output_folder, samples, iteration):
        def build_response(key, val):
            def response(x):
                return val.response({
                    'sky': sky_dict['sky'](x),
                    '_'.join((key, 'zero_flux_mean')): x['_'.join((key, 'zero_flux_mean'))]
                })

            return response

        fig, axes = plt.subplots(
            len(data), 3, figsize=(3*3, len(data)*3), dpi=300)

        for ii, (key, val) in enumerate(data.items()):
            d = val.data_2d
            n = val.noise_2d
            response = build_response(key, val)
            sky = jft.mean([response(s) for s in samples])
            res = (d - sky)/n

            im0 = axes[ii, 0].imshow(d, origin='lower', norm=norm)
            im1 = axes[ii, 1].imshow(sky, origin='lower', norm=norm)
            im2 = axes[ii, 2].imshow(
                res, origin='lower', cmap='RdBu_r', vmin=-3, vmax=3)

            for im, ax in zip([im0, im1, im2], axes[ii]):
                fig.colorbar(im, ax=ax, shrink=0.7)
            axes[ii, 0].set_title(key)
            axes[ii, 1].set_title('R(sky)')
            axes[ii, 2].set_title('(d-R(sky))/n')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"residuals_{iteration}.png"))
        plt.close()

    return plot_simple_residuals


if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

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

    residual_dict = {key: build_residuals(key, val)
                     for key, val in data.items()}
    plot_simple_residuals = build_plot_simple_residuals(data, log_data=True)

    def plot(s, x):
        if False:
            ju.plot_sample_and_stats(file_info["res_dir"],
                                     residual_dict,
                                     s,
                                     log_scale=False,
                                     relative_std=True,
                                     iteration=x.nit,
                                     plotting_kwargs=dict(
                                         vmin=-1, vmax=1, cmap='RdBu_r')
                                     )

        ju.plot_sample_and_stats(file_info["res_dir"],
                                 sky_dict,
                                 s,
                                 log_scale=True,
                                 relative_std=True,
                                 iteration=x.nit)

        ju.export_operator_output_to_fits(file_info["res_dir"],
                                          sky_dict,
                                          s,
                                          iteration=x.nit)

        plot_simple_residuals(file_info["res_dir"], s, x.nit)

    samples, state = jft.optimize_kl(log_likelihood,
                                     pos_init,
                                     key=key,
                                     kl_kwargs=kl_solver_kwargs,
                                     callback=plot,
                                     odir=file_info["res_dir"],
                                     **minimization_config)
