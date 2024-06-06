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
from jubik0.jwst.jwst_plotting import build_plot, build_plot_lens_light
from jubik0.jwst.filter_projector import FilterProjector

from charm_lensing import minimization_parser
from charm_lensing import build_lens_system

from sys import exit

if True:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])

config_path = './demos/lens_config.yaml'
cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
RES_DIR = cfg['files']['res_dir']
# FIXME: This needs to provided somewhere else
DATA_DVOL = (0.13*u.arcsec**2).to(u.deg**2)

reconstruction_grid = build_reconstruction_grid_from_config(cfg)

lens_system = build_lens_system.build_lens_system(cfg['lensing'])
sky_model = lens_system.get_forward_model_parametric()


filter_projector = FilterProjector(
    sky_domain=sky_model.target,
    key_and_index={key: ii for ii, key in enumerate(cfg['grid']['e_keys'])}
)
sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)),
    init=sky_model.init
)

for ii in range(4):
    key, test_key = random.split(random.PRNGKey(42+ii), 2)
    x = jft.random_like(test_key, sky_model.domain)

    _sky_model = lens_system.source_plane_model.light_model.nonparametric().model._sky_model
    alpha = _sky_model.alpha_cf(x)
    plaw = _sky_model.plaw(x)

    sky_model_keys = sky_model_with_keys.target.keys()
    lens_light_alpha, lens_light_full = (
        lens_system.lens_plane_model.light_model.nonparametric().model._sky_model.alpha_cf,
        lens_system.lens_plane_model.light_model)
    source_light_alpha, source_light_full = (
        lens_system.source_plane_model.light_model.nonparametric().model._sky_model.alpha_cf,
        lens_system.source_plane_model.light_model)
    lensed_light = lens_system.get_forward_model_parametric(only_source=True)

    xlen = len(sky_model_keys) + 1
    fig, axes = plt.subplots(3, xlen, figsize=(3*xlen, 8), dpi=300)
    ims = np.zeros_like(axes)

    # Plot lens light
    ims[0, 0] = axes[0, 0].imshow(lens_light_alpha(x), origin='lower')
    leli = lens_light_full(x)
    for ii, filter_name in enumerate(sky_model_keys):
        axes[0, ii+1].set_title(f'Lens light {filter_name}')
        ims[0, ii+1] = axes[0, ii+1].imshow(
            leli[ii], origin='lower',
            norm=LogNorm(vmin=np.max((1e-5, leli.min())), vmax=leli.max()))

    # PLOT lensed light
    lsli = lensed_light(x)
    ims[1, 0] = axes[1, 0].imshow(
        (leli+lsli)[0], origin='lower',
        norm=LogNorm(vmin=np.max((1e-5, (leli+lsli)[0].min())),
                     vmax=(leli+lsli)[0].max()))
    for ii, filter_name in enumerate(sky_model_keys):
        axes[1, ii+1].set_title(f'Lensed light {filter_name}')
        ims[1, ii+1] = axes[1, ii+1].imshow(
            lsli[ii], origin='lower',
            norm=LogNorm(vmin=np.max((1e-5, lsli.min())), vmax=lsli.max()))

    # Plot lens light
    ims[2, 0] = axes[2, 0].imshow(source_light_alpha(x), origin='lower')
    slli = source_light_full(x)
    for ii, filter_name in enumerate(sky_model_keys):
        axes[2, ii+1].set_title(f'Lens light {filter_name}')
        ims[2, ii+1] = axes[2, ii+1].imshow(
            slli[ii], origin='lower',
            vmin=slli.min(), vmax=slli.max())

    for ax, im in zip(axes.flatten(), ims.flatten()):
        if not isinstance(im, int):
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    plt.show()

data_plotting = {}
likelihoods = []
kk = 0
for fltname, flt in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        jwst_data = JwstData(filepath)

        data_key = f'{fltname}_{ii}'

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
            {fltname: sky_model_with_keys.target[fltname]},

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
                ext=(psf_ext, psf_ext))
        )

        data_plotting[data_key] = dict(
            index=filter_projector.key_and_index[fltname],
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


key = random.PRNGKey(87)
key, rec_key = random.split(key, 2)


cfg_mini = ju.get_config('demos/lens_config.yaml')["minimization"]
key = random.PRNGKey(cfg_mini.get('key', 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

n_samples = minimization_parser.n_samples_factory(cfg_mini)
mode_samples = minimization_parser.sample_type_factory(cfg_mini)
linear_kwargs = minimization_parser.linear_sample_kwargs_factory(cfg_mini)
nonlin_kwargs = minimization_parser.nonlinear_sample_kwargs_factory(cfg_mini)
kl_kwargs = minimization_parser.kl_kwargs_factory(cfg_mini)

# minimization_config = cfg_mini['minimization']
# kl_solver_kwargs = minimization_config.pop('kl_kwargs')
# minimization_config['resume'] = True
# minimization_config['n_samples'] = lambda it: 4 if it < 10 else 10


plot_sky = build_plot_lens_light(
    results_directory=RES_DIR,
    sky_model_keys=sky_model_with_keys.target.keys(),
    lens_light=(
        lens_system.lens_plane_model.light_model.nonparametric().model._sky_model.alpha_cf,
        lens_system.lens_plane_model.light_model),
    source_light=(
        lens_system.source_plane_model.light_model.nonparametric().model._sky_model.alpha_cf,
        lens_system.source_plane_model.light_model),
    lensed_light=lens_system.get_forward_model_parametric(only_source=True),
    plotting_config=dict(
        norm_lens=LogNorm,
        # norm_source=LogNorm,
    )
)


residual_plot = build_plot(
    data_dict=data_plotting,
    sky_model_with_key=sky_model_with_keys,
    sky_model=sky_model,
    small_sky_model=lens_system.source_plane_model.light_model,
    results_directory=RES_DIR,
    alpha=lens_system.source_plane_model.light_model.nonparametric().model._sky_model.alpha_cf,
    plotting_config=dict(
        norm=LogNorm,
        sky_extent=None,
        plot_sky=False
    ))


def plot_new(samples, state):
    residual_plot(samples, state)
    plot_sky(samples, state)


print(f'Results: {RES_DIR}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,

    callback=plot_new,
    odir=RES_DIR,

    n_total_iterations=cfg_mini['n_total_iterations'],
    n_samples=n_samples,
    sample_mode=mode_samples,
    draw_linear_kwargs=linear_kwargs,
    nonlinearly_update_kwargs=nonlin_kwargs,
    kl_kwargs=kl_kwargs,
    resume=cfg_mini.get('resume', False),
)
