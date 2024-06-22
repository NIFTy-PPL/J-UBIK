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
from jubik0.jwst.config_handler import build_reconstruction_grid_from_config
from jubik0.jwst.wcs import (subsample_grid_centers_in_index_grid)
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.jwst.jwst_plotting import build_plot
from jubik0.jwst.filter_projector import FilterProjector

from jubik0.jwst.color import Color, ColorRange

from charm_lensing import minimization_parser

import os
from os.path import join
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
    join(RES_DIR, 'hashes.txt'))

# FIXME: This needs to provided somewhere else
DATA_DVOL = (0.13*u.arcsec**2).to(u.deg**2)

reconstruction_grid = build_reconstruction_grid_from_config(cfg)

if cfg['priors']['diffuse'].get('colormix'):
    from copy import deepcopy
    energy_bins = cfg['grid'].get('edim')

    energy_cfg = cfg['grid'].get('energy_bin')
    diffuse_priors = cfg['priors']['diffuse']

    components_prior_config = {
        f'k{ii}': deepcopy(diffuse_priors['spatial']) for ii in range(energy_bins)}

    components_config = dict(
        shape=reconstruction_grid.shape,
        distances=[
            d.to(u.arcsec).value for d in reconstruction_grid.distances],
        s_padding_ratio=cfg['grid'].get('s_padding_ratio', 1.0),
        prior=components_prior_config,
    )

    small_sky_model = sky_model = ju.build_colormix_components(
        'sky',
        colormix_config=diffuse_priors['colormix'],
        components_config=components_config)

    def alpha(x):
        return jnp.ones((10, 10))

    ju.prior_samples_colormix_components(sky_model, 4)

else:
    sky_model_new = ju.SkyModel(config_file_path=config_path)
    small_sky_model = sky_model_new.create_sky_model(fov=cfg['grid']['fov'])
    sky_model = sky_model_new.full_diffuse
    energy_cfg = sky_model_new.config['grid']['energy_bin']

    alpha = sky_model_new.alpha_cf


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

data_plotting = {}
likelihoods = []
kk = 0
for fltname, flt_dct in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt_dct):
        print(fltname, ii, filepath)
        jwst_data = JwstData(filepath)
        ekey = filter_projector.get_key(jwst_data.pivot_wavelength)

        data_key = f'{fltname}_{ekey}_{ii}'

        # FIXME: This can also be handled by passing a delta for the priors
        # of the shift, and rotation
        # FIXME: The creation of the correction_model should be moved in the
        # build_rotation_and_shift_model inside build_data_model.
        # This will remove the coords partial over-write inside the
        # build_rotation_and_shift_model. Once this is done only the prior for
        # the rotation and shift correction will be passed to the
        # shift_and_rotation kwargs dictionary.
        if kk == 0:
            correction_model = None
        else:
            from jubik0.jwst.rotation_and_shift.coordinates_correction import build_coordinates_correction_model_from_grid
            correction_model = build_coordinates_correction_model_from_grid(
                domain_key=data_key + '_correction',
                priors=dict(
                    shift=('normal', 0, 1.0e-1),
                    rotation=('normal', 0, (1.0e-1*u.deg).to(u.rad).value)),
                data_wcs=jwst_data.wcs,
                reconstruction_grid=reconstruction_grid,
            )

        kk += 1

        psf_ext = int(cfg['telescope']['psf']['psf_pixels'] // 2)
        # define a mask
        data_centers = np.squeeze(subsample_grid_centers_in_index_grid(
            reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)),
            jwst_data.wcs,
            reconstruction_grid.wcs,
            1))
        mask = get_mask_from_index_centers(
            data_centers, reconstruction_grid.shape)
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
                data_dvol=DATA_DVOL,
                data_wcs=jwst_data.wcs,
                data_model_type=cfg['telescope']['rotation_and_shift']['model'],
                shift_and_rotation_correction=correction_model,
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

            data_mask=mask,

            world_extrema=reconstruction_grid.world_extrema(
                ext=(psf_ext, psf_ext)),

            zero_flux=dict(
                dkey=data_key,
                zero_flux=dict(prior=('lognormal', 1, 3))
            ))

        data_plotting[data_key] = dict(
            index=filter_projector.keys_and_index[ekey],
            data=data,
            std=std,
            mask=mask,
            data_model=data_model,
            correction_model=correction_model)

        likelihood = build_gaussian_likelihood(
            jnp.array(data[mask], dtype=float),
            jnp.array(std[mask], dtype=float))
        likelihood = likelihood.amend(
            data_model, domain=jft.Vector(data_model.domain))
        likelihoods.append(likelihood)


model = sky_model_with_keys
likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(likelihood, model)


cfg_mini = ju.get_config('demos/jwst_config.yaml')["minimization"]
key = random.PRNGKey(cfg_mini.get('key', 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

n_samples = minimization_parser.n_samples_factory(cfg_mini)
mode_samples = minimization_parser.sample_type_factory(cfg_mini)
linear_kwargs = minimization_parser.linear_sample_kwargs_factory(cfg_mini)
nonlin_kwargs = minimization_parser.nonlinear_sample_kwargs_factory(cfg_mini)
kl_kwargs = minimization_parser.kl_kwargs_factory(cfg_mini)


plot = build_plot(
    data_dict=data_plotting,
    sky_model_with_key=sky_model_with_keys,
    sky_model=sky_model,
    small_sky_model=small_sky_model,
    results_directory=RES_DIR,
    alpha=alpha,
    plotting_config=dict(
        norm=LogNorm,
        sky_extent=None,
        plot_sky=False
    ))


print(f'Results: {RES_DIR}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,

    callback=plot,
    odir=RES_DIR,

    n_total_iterations=cfg_mini['n_total_iterations'],
    n_samples=n_samples,
    sample_mode=mode_samples,
    draw_linear_kwargs=linear_kwargs,
    nonlinearly_update_kwargs=nonlin_kwargs,
    kl_kwargs=kl_kwargs,
    resume=cfg_mini.get('resume', False),
)
