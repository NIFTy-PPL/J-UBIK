import yaml
from functools import reduce

import nifty8.re as jft
from jax import random
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import units as u

import jubik0 as ju
from jubik0.library.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)
from jubik0.jwst.jwst_data import JwstData
from jubik0.jwst.masking import get_mask_from_index_centers
from jubik0.jwst.config_handler import (
    build_reconstruction_grid_from_config,
    build_coordinates_correction_prior_from_config,
    insert_ubik_energy_in_lensing,
    insert_spaces_in_lensing)
from jubik0.jwst.wcs import (subsample_grid_centers_in_index_grid)
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.jwst.jwst_plotting import (
    build_plot, build_color_components_plotting, build_plot_lens_system, get_alpha_nonpar)
from jubik0.jwst.filter_projector import FilterProjector

from jubik0.jwst.color import Color, ColorRange

from charm_lensing import build_lens_system

from sys import exit
import os


config_path = './demos/jwst_lens_config.yaml'
cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
RES_DIR = cfg['files']['res_dir']
os.makedirs(RES_DIR, exist_ok=True)
ju.save_local_packages_hashes_to_txt(
    ['nifty8', 'charm_lensing', 'jubik0'],
    os.path.join(RES_DIR, 'hashes.txt'))
ju.save_config_copy('jwst_lens_config.yaml', './demos', RES_DIR)

if cfg['cpu']:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])


reconstruction_grid = build_reconstruction_grid_from_config(cfg)
insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing(cfg)
lens_system = build_lens_system.build_lens_system(cfg['lensing'])
sky_model = lens_system.get_forward_model_parametric()


energy_cfg = cfg['grid']['energy_bin']
e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
keys_and_colors = {
    f'e{ii:02d}': ColorRange(Color(emin*e_unit), Color(emax*e_unit))
    for ii, (emin, emax) in enumerate(zip(energy_cfg.get('e_min'), energy_cfg.get('e_max')))}

filter_projector = FilterProjector(
    sky_domain=sky_model.target,
    keys_and_colors=keys_and_colors,
)
sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)),
    init=sky_model.init
)

data_dict = {}
likelihoods = []
for fltname, flt in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        jwst_data = JwstData(filepath)
        ekey = filter_projector.get_key(jwst_data.pivot_wavelength)

        data_key = f'{fltname}_{ekey}_{ii}'

        # Loading data, std, and mask.
        psf_ext = int(cfg['telescope']['psf']['psf_pixels'] // 2)
        mask = get_mask_from_index_centers(
            np.squeeze(subsample_grid_centers_in_index_grid(
                reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)),
                jwst_data.wcs,
                reconstruction_grid.wcs,
                1)),
            reconstruction_grid.shape)
        mask *= jwst_data.nan_inside_extrema(
            reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)))
        data = jwst_data.data_inside_extrema(
            reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)))
        std = jwst_data.std_inside_extrema(
            reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)))

        data_model = build_data_model(
            {ekey: sky_model_with_keys.target[ekey]},
            reconstruction_grid=reconstruction_grid,
            subsample=cfg['telescope']['rotation_and_shift']['subsample'],

            rotation_and_shift_kwargs=dict(
                data_dvol=jwst_data.dvol,
                data_wcs=jwst_data.wcs,
                data_model_type=cfg['telescope']['rotation_and_shift']['model'],
                shift_and_rotation_correction=dict(
                    domain_key=data_key + '_correction',
                    priors=build_coordinates_correction_prior_from_config(
                        cfg, jwst_data.filter, ii),
                )
            ),

            psf_kwargs=dict(
                camera=jwst_data.camera,
                filter=jwst_data.filter,
                center_pixel=jwst_data.wcs.index_from_wl(
                    reconstruction_grid.center)[0],
                webbpsf_path=cfg['telescope']['psf']['webbpsf_path'],
                psf_library_path=cfg['telescope']['psf']['psf_library'],
                fov_pixels=cfg['telescope']['psf']['psf_pixels'],
            ),

            transmission=jwst_data.transmission,

            data_mask=mask,

            world_extrema=reconstruction_grid.world_extrema(
                ext=(psf_ext, psf_ext)),

            zero_flux=dict(
                dkey=data_key,
                zero_flux=dict(prior=cfg['telescope']['zero_flux']['prior']),
            ),
        )

        data_dict[data_key] = dict(
            index=filter_projector.keys_and_index[ekey],
            data=data,
            std=std,
            mask=mask,
            data_model=data_model,
        )

        likelihood = build_gaussian_likelihood(
            jnp.array(data[mask], dtype=float),
            jnp.array(std[mask], dtype=float))
        likelihood = likelihood.amend(
            data_model, domain=jft.Vector(data_model.domain))
        likelihoods.append(likelihood)


model = sky_model_with_keys
likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(likelihood, model)

# PLOTTING
plot_components_switch = hasattr(
    lens_system.source_plane_model.light_model.nonparametric(), 'color')
parametric_flag = lens_system.lens_plane_model.convergence_model.nonparametric() is not None

ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(
    lens_system, plot_components_switch)

plot_lens_light = build_plot_lens_system(
    RES_DIR,
    plotting_config=dict(
        # norm_source=LogNorm,
        norm_lens=LogNorm,
        # norm_mass=LogNorm,
    ),
    lens_system=lens_system,
    filter_projector=filter_projector,
    lens_light_alpha_nonparametric=(ll_alpha, ll_nonpar),
    source_light_alpha_nonparametric=(sl_alpha, sl_nonpar),
)

mass_model = lens_system.lens_plane_model.convergence_model.parametric()
residual_plot = build_plot(
    data_dict=data_dict,
    sky_model_with_key=sky_model_with_keys,
    sky_model=sky_model,
    small_sky_model=lens_system.get_forward_model_parametric(),
    results_directory=RES_DIR,
    upperleft=('Mass field', mass_model),
    plotting_config=dict(
        norm=LogNorm,
        sky_extent=None,
        plot_sky=False
    ))


if plot_components_switch:
    plot_color = build_color_components_plotting(
        lens_system.source_plane_model.light_model.nonparametric(), RES_DIR)


if cfg.get('prior_samples') is not None:
    test_key, _ = random.split(random.PRNGKey(42), 2)
    for ii in range(cfg.get('prior_samples', 3)):
        test_key, _ = random.split(test_key, 2)
        position = jft.random_like(test_key, model.domain)

        key, test_key = random.split(test_key, 2)
        if plot_components_switch:
            pass

        else:
            plot_lens_light(position, None, parametric=parametric_flag)

            for ii, (dkey, data) in enumerate(data_dict.items()):
                dm = data['data_model']
                dd = data['data']
                std = data['std']
                mask = data['mask']


def plot(samples, state):
    residual_plot(samples, state)
    plot_lens_light(samples, state, parametric=parametric_flag)
    if plot_components_switch:
        plot_color(samples, state)


cfg_mini = ju.get_config('demos/jwst_lens_config.yaml')["minimization"]
n_dof = ju.calculate_n_constrained_dof(likelihood)
minpars = ju.MinimizationParser(cfg_mini, n_dof)
key = random.PRNGKey(cfg_mini.get('key', 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))


print(f'Results: {RES_DIR}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,

    callback=plot,
    odir=RES_DIR,

    n_total_iterations=cfg_mini['n_total_iterations'],
    n_samples=minpars.n_samples,
    sample_mode=minpars.sample_mode,
    draw_linear_kwargs=minpars.draw_linear_kwargs,
    nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
    kl_kwargs=minpars.kl_kwargs,
    resume=cfg_mini.get('resume', False),
)
