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
from jubik0.jwst.wcs import (subsample_grid_centers_in_index_grid)
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.jwst.jwst_plotting import build_plot, build_plot_lens_light
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

if cfg['cpu']:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])


reconstruction_grid = build_reconstruction_grid_from_config(cfg)
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
kk = 0
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
                        kk, cfg),
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

        kk += 1


for ii in range(0):

    # import os
    # opath = 'results/jwst_test/mf_f356w_f444w_lens_06'
    # compsky = last_fn = os.path.join(opath, 'last.pkl')

    key, test_key = random.split(random.PRNGKey(42+ii), 2)
    x = jft.random_like(test_key, sky_model.domain)

    _sky_model = lens_system.source_plane_model.light_model.nonparametric()._sky_model
    alpha = _sky_model.alpha_cf(x)
    plaw = _sky_model.plaw(x)

    sky_model_keys = sky_model_with_keys.target.keys()
    lens_light_alpha, lens_light_full = (
        lens_system.lens_plane_model.light_model.nonparametric()._sky_model.alpha_cf,
        lens_system.lens_plane_model.light_model)
    source_light_alpha, source_light_full = (
        lens_system.source_plane_model.light_model.nonparametric()._sky_model.alpha_cf,
        lens_system.source_plane_model.light_model)
    lensed_light = lens_system.get_forward_model_parametric(only_source=True)

    xlen = max(len(sky_model_keys) + 1, 3)
    fig, axes = plt.subplots(
        3 + len(keys_and_colors.keys()), xlen, figsize=(3*xlen*3, 8*3), dpi=300)
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
        axes[2, ii+1].set_title(f'Source light {filter_name}')
        ims[2, ii+1] = axes[2, ii+1].imshow(
            slli[ii], origin='lower',
            vmin=slli.min(), vmax=slli.max())

    flight = filter_projector(leli + lsli)

    color_indices = []
    jj = 0
    for (dkey, valdict) in data_dict.items():
        if valdict['index'] in color_indices:
            continue
        color_indices.append(valdict['index'])
        print(dkey, jj)

        data, std, mask, data_model = (
            valdict['data'], valdict['std'], valdict['mask'], valdict['data_model'])

        latent_position = jft.random_like(test_key, data_model.domain)
        for ckey, cval in flight.items():
            latent_position[ckey] = cval
        rs = np.zeros(mask.shape)
        rs[mask] = data_model(latent_position)

        axes[jj+3, 0].imshow(data, origin='lower', norm=LogNorm())
        axes[jj+3, 1].imshow(rs, origin='lower', norm=LogNorm())
        axes[jj+3, 2].imshow(data-rs, origin='lower', cmap='RdBu_r')
        axes[jj+3, 2].set_title('data-data_model')

        jj += 1

    for ax, im in zip(axes.flatten(), ims.flatten()):
        if not isinstance(im, int):
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    plt.show()


model = sky_model_with_keys
likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(likelihood, model)


tmp_alpha = lens_system.lens_plane_model.light_model.nonparametric()._sky_model.alpha_cf
ll_shape = lens_system.lens_plane_model.space.shape
ll_alpha = jft.Model(
    lambda x: tmp_alpha(x)[:ll_shape[0], :ll_shape[1]],
    domain=tmp_alpha.domain)

ll_nonpar = lens_system.lens_plane_model.light_model.parametric(
).nonparametric()[0].nonparametric()
if ll_nonpar is None:
    def ll_nonpar(_): return jnp.zeros((12, 12))

sl_nonpar = lens_system.source_plane_model.light_model.nonparametric()._sky_model.spatial_cf
if sl_nonpar is None:
    def sl_nonpar(_): return jnp.zeros((12, 12))

mass_model = lens_system.lens_plane_model.convergence_model.parametric()

plot_sky = build_plot_lens_light(
    results_directory=RES_DIR,
    sky_model_keys=sky_model_with_keys.target.keys(),
    lens_light=(ll_alpha, lens_system.lens_plane_model.light_model, ll_nonpar),
    source_light=(
        lens_system.source_plane_model.light_model.nonparametric()._sky_model.alpha_cf,
        lens_system.source_plane_model.light_model, sl_nonpar),
    lensed_light=lens_system.get_forward_model_parametric(only_source=True),
    mass_model=mass_model,
    plotting_config=dict(
        norm_lens=LogNorm,
        # norm_source=LogNorm,
    )
)

residual_plot = build_plot(
    data_dict=data_dict,
    sky_model_with_key=sky_model_with_keys,
    sky_model=sky_model,
    small_sky_model=lens_system.get_forward_model_parametric(),
    results_directory=RES_DIR,
    alpha=ll_alpha,
    plotting_config=dict(
        norm=LogNorm,
        sky_extent=None,
        plot_sky=False
    ))


def plot(samples, state):
    plot_sky(samples, state)
    residual_plot(samples, state)


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
