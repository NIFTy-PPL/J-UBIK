import argparse
import os
from functools import reduce
from os.path import join

import astropy.units as u
from astropy.coordinates import SkyCoord

import nifty8.re as jft
import numpy as np
from jax import config, random

import jubik0 as ju
from jubik0.library.instruments.jwst.filter_projector import FilterProjector
from jubik0.library.instruments.jwst.likelihood import (
    build_gaussian_likelihood, connect_likelihood_to_model)

config.update('jax_enable_x64', True)


# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help="Config file (.yaml) for JWST inference.",
                    nargs='?', const=1, default="configs/jwst_demo.yaml")
args = parser.parse_args()

filter_distance = dict(
    f115w=0.031,
    f150w=0.031,
    f200w=0.031,
)

if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

    # Uncomment to save local packages git hashes to file
    # ju.save_local_packages_hashes_to_txt(
    #     ['jubik0', 'nifty8'],
    #     os.path.join(file_info['res_dir'], "packages_hashes.txt"),
    #     paths_to_git=[os.path.dirname(os.getcwd()), None],
    #     verbose=False)

    # Save run configuration
    ju.copy_config(os.path.basename(config_path),
                   output_dir=file_info['res_dir'])

    # Load sky model
    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    # Draw random numbers
    key = random.PRNGKey(cfg['seed'])

    energy_cfg = cfg['grid']['energy_bin']
    e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
    filters = {'f115w': 2, 'f150w': 1, 'f200w': 0}
    filter_projector = FilterProjector(
        sky_domain=sky.target,
        keys_and_colors=filters,
    )

    key, subkey = random.split(key)
    sky_model_with_filters = jft.Model(
        lambda x: filter_projector(sky(x)),
        domain=sky.domain)

    mock_sky = sky_model_with_filters(sky_model_with_filters.init(subkey))
    center = (0, 0)
    reconstruction_grid = ju.Grid(
        center=SkyCoord(center[0]*u.rad, center[1]*u.rad),
        shape=(cfg['grid']['sdim'],)*2,
        fov=(cfg['grid']['fov']*u.arcsec,)*2
    )

    data_set = {}
    likelihoods = []
    for fltname in filters.keys():

        psf_kwargs = dict(
            camera='nircam',
            filter=fltname,
            center_pixel=cfg['telescope']['pointing_pixel'],
            webbpsf_path=file_info['webbpsf_path'],
            psf_library_path=file_info['psf_library'],
            fov_pixels=cfg['telescope']['fov'],
        )

        data_fov = cfg['telescope']['fov']
        data_shape = int(data_fov/filter_distance[fltname])
        data_grid = ju.Grid(
            center=SkyCoord(center[0]*u.rad, center[1]*u.rad),
            shape=(data_shape,)*2,
            fov=(data_fov*u.arcsec,)*2
        )
        rotation_and_shift_kwargs = dict(
            reconstruction_grid=reconstruction_grid,
            data_dvol=data_grid.dvol,
            data_wcs=data_grid.wcs,
            data_model_type='linear',
            world_extrema=data_grid.world_extrema(),
        )

        data_model = ju.build_jwst_data_model(
            sky_domain={fltname: filter_projector.target[fltname]},
            subsample=cfg['telescope']['subsample'],
            rotation_and_shift_kwargs=rotation_and_shift_kwargs,
            psf_kwargs=psf_kwargs,
            data_mask=None,
            transmission=1.,
            zero_flux=None,
        )

        # Create mock data
        noise_std = cfg['mock_config']['noise_std']
        key, subkey = random.split(key)
        data = (
            data_model(mock_sky) +
            random.normal(subkey, data_model.target.shape) * noise_std
        )

        data_set[fltname] = {
            'data': data,
            'std': noise_std,
            'data_model': data_model,
            'data_grid': data_grid,
        }

        likelihood = build_gaussian_likelihood(data, noise_std)
        likelihood = likelihood.amend(
            data_model, domain=jft.Vector(data_model.domain))
        likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x+y, likelihoods)
    likelihood = connect_likelihood_to_model(
        likelihood, sky_model_with_filters)

    data = np.array([d["data"] for d in data_set.values()])
    ju.plot_result(data, output_file=join(file_info["res_dir"], "data.png"))

    # Plot
    additional_plot_dict = {}
    if hasattr(sky_model, 'alpha_cf'):
        additional_plot_dict['diffuse_alfa'] = sky_model.alpha_cf
    if hasattr(sky_model, 'points_alfa'):
        additional_plot_dict['points_alfa'] = sky_model.points_alfa

    def simple_eval_plots(s, x):
        """Call plot_sample_and_stat for every iteration."""
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 sky_dict,
                                 s,
                                 dpi=cfg["plotting"]["dpi"],
                                 iteration=x.nit,
                                 rgb_min_sat=[3e-8, 3e-8, 3e-8],
                                 rgb_max_sat=[2.0167e-6, 1.05618e-6, 1.5646e-6])
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 additional_plot_dict,
                                 s,
                                 dpi=cfg["plotting"]["dpi"],
                                 iteration=x.nit,
                                 log_scale=False,
                                 plot_samples=False,
                                 )
        ju.plot_pspec(sky_model.spatial_pspec,
                      sky_model.spatial_cf.target.shape,
                      sky_model.s_distances,
                      s,
                      file_info["res_dir"],
                      iteration=x.nit,
                      dpi=cfg["plotting"]["dpi"],
                      )

    # Minimization
    minimization_config = cfg['minimization']
    n_dof = ju.get_n_constrained_dof(likelihood)
    minimization_parser = ju.MinimizationParser(minimization_config,
                                                n_dof=n_dof)
    key, subkey = random.split(key)
    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    samples, state = jft.optimize_kl(
        likelihood,
        pos_init,
        key=key,
        n_total_iterations=minimization_config['n_total_iterations'],
        resume=minimization_config['resume'],
        n_samples=minimization_parser.n_samples,
        draw_linear_kwargs=minimization_parser.draw_linear_kwargs,
        nonlinearly_update_kwargs=minimization_parser.nonlinearly_update_kwargs,
        kl_kwargs=minimization_parser.kl_kwargs,
        sample_mode=minimization_parser.sample_mode,
        callback=simple_eval_plots,
        odir=file_info["res_dir"], )