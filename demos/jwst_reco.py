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
    build_coordinates_correction_prior_from_config)
from jubik0.jwst.utils import build_sky_model_from_config
from jubik0.jwst.wcs import subsample_grid_centers_in_index_grid
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.jwst.jwst_plotting import (
    build_plot_sky_residuals, build_color_components_plotting)
from jubik0.jwst.filter_projector import FilterProjector

from jubik0.jwst.color import Color, ColorRange

import os
from sys import exit

if False:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])

config_path = './demos/jwst_config.yaml'
cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)

RES_DIR = cfg['files']['res_dir']
os.makedirs(RES_DIR, exist_ok=True)
ju.save_local_packages_hashes_to_txt(
    ['nifty8', 'charm_lensing', 'jubik0'],
    os.path.join(RES_DIR, 'hashes.txt'))
ju.save_config_copy('jwst_config.yaml', './demos', RES_DIR)


if cfg['cpu']:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])


reconstruction_grid = build_reconstruction_grid_from_config(cfg)
small_sky_model, sky_model, alpha, energy_cfg = build_sky_model_from_config(
    cfg, reconstruction_grid)


assert cfg['grid']['edim'] == len(cfg['grid']['energy_bin']['e_min'])
assert cfg['grid']['edim'] == len(cfg['grid']['energy_bin']['e_max'])
e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
keys_and_colors = {
    f'e{ii:02d}': ColorRange(Color(emin*e_unit), Color(emax*e_unit))
    for ii, (emin, emax) in
    enumerate(zip(energy_cfg.get('e_min'), energy_cfg.get('e_max')))
}

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
for fltname, flt_dct in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt_dct):
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
                        cfg, jwst_data.filter, ii),  # could also be None
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


residual_plot = build_plot_sky_residuals(
    results_directory=RES_DIR,
    data_dict=data_dict,
    sky_model_with_key=sky_model_with_keys,
    small_sky_model=small_sky_model,
    overwrite_model=((-1, 0), 'Alpha field', alpha),
    plotting_config=dict(
        norm=LogNorm,
        sky_extent=None,
        plot_sky=False
    ))
plot_color = build_color_components_plotting(sky_model, RES_DIR)


if cfg.get('prior_samples') is not None:
    test_key, _ = random.split(random.PRNGKey(42), 2)
    for ii in range(cfg.get('prior_samples', 3)):
        test_key, _ = random.split(test_key, 2)
        position = jft.random_like(test_key, likelihood.domain)
        while isinstance(position, jft.Vector):
            position = position.tree
        residual_plot(position)
        plot_color(position)


def plot(samples, x):
    residual_plot(samples, x)
    plot_color(samples, x)


cfg_mini = ju.get_config('demos/jwst_config.yaml')["minimization"]
key = random.PRNGKey(cfg_mini.get('key', 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))
n_dof = ju.calculate_n_constrained_dof(likelihood)
minpars = ju.MinimizationParser(cfg_mini, n_dof)

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


for ii, (data_key, data_set) in enumerate(data_dict.items()):
    data_model = data_set['data_model']
    rotation_trafo = data_model.rotation_and_shift.correction_model.rotation_prior
    shift_trafo = data_model.rotation_and_shift.correction_model.shift_prior

    shift = jft.mean([shift_trafo(x) for x in samples])
    rotation = jft.mean([rotation_trafo(x) for x in samples])

    print(data_key, f'shift: [{shift[0]}, {shift[1]}]',
          f'rotation: {rotation}')
