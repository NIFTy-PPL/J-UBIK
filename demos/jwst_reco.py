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


from sys import exit


config_path = './demos/JWST_config.yaml'
cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
RES_DIR = cfg['files']['res_dir']
D_PADDING_RATIO = cfg['grid']['d_padding_ratio']
# FIXME: This needs to provided somewhere else
DATA_DVOL = (0.13*u.arcsec**2).to(u.deg**2)
FOV_PIXELS = 32

reconstruction_grid = build_reconstruction_grid_from_config(cfg)

sky_model_new = ju.SkyModel(config_file_path=config_path)
small_sky_model = sky_model_new.create_sky_model(fov=cfg['grid']['fov'])
sky_model = sky_model_new.full_diffuse


key = random.PRNGKey(87)
key, test_key, rec_key = random.split(key, 3)


filter_projector = FilterProjector(
    sky_domain=sky_model.target,
    key_and_index={key: ii for ii, key in enumerate(cfg['grid']['e_keys'])}
)
sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)),
    init=sky_model.init
)


data_plotting = {}
likelihoods = []
for fltname, flt in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        jwst_data = JwstData(filepath)
        # print(jwst_data.dm.meta.date)

        data_key = f'{fltname}_{ii}'

        # define a mask
        data_centers = np.squeeze(subsample_grid_centers_in_index_grid(
            reconstruction_grid.world_extrema(D_PADDING_RATIO),
            jwst_data.wcs,
            reconstruction_grid.wcs,
            1))
        mask = get_mask_from_index_centers(
            data_centers, reconstruction_grid.shape)
        mask *= jwst_data.nan_inside_extrema(
            reconstruction_grid.world_extrema(D_PADDING_RATIO))

        data = jwst_data.data_inside_extrema(
            reconstruction_grid.world_extrema(D_PADDING_RATIO))
        std = jwst_data.std_inside_extrema(
            reconstruction_grid.world_extrema(D_PADDING_RATIO))

        data_model = build_data_model(
            {fltname: sky_model_with_keys.target[fltname]},

            reconstruction_grid=reconstruction_grid,

            subsample=cfg['telescope']['rotation_and_shift']['subsample'],

            rotation_and_shift_kwargs=dict(
                data_dvol=DATA_DVOL,
                data_wcs=jwst_data.wcs,
                data_model_type=cfg['telescope']['rotation_and_shift']['model'],
            ),

            psf_kwargs=dict(
                camera=jwst_data.camera,
                filter=jwst_data.filter,
                center_pixel=jwst_data.wcs.index_from_wl(
                    reconstruction_grid.center)[0],
                webbpsf_path=cfg['telescope']['psf']['webbpsf_path'],
                psf_library_path=cfg['telescope']['psf']['psf_library'],
                fov_pixels=FOV_PIXELS,
            ),

            data_mask=mask,

            world_extrema=reconstruction_grid.world_extrema(D_PADDING_RATIO)
        )

        data_plotting[data_key] = dict(
            data=data,
            std=std,
            mask=mask,
            data_model=data_model)

        likelihood = build_gaussian_likelihood(
            jnp.array(data[mask], dtype=float),
            jnp.array(std[mask], dtype=float))
        likelihood = likelihood.amend(
            data_model, domain=jft.Vector(data_model.domain))
        likelihoods.append(likelihood)

likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(
    likelihood,
    sky_model_with_keys
)


key = random.PRNGKey(87)
key, rec_key = random.split(key, 2)

for ii in range(3):
    key, test_key = random.split(key, 2)
    x = jft.random_like(test_key, sky_model.domain)
    sky = sky_model_with_keys(x)
    # plot_sky(sky, data_plotting)

    plaw = sky_model_new.plaw(x)
    alpha = sky_model_new.alpha_cf(x)

    fig, axes = plt.subplots(len(sky)+1, 4)
    integrated_sky = []
    for ii, (axi, sky_key) in enumerate(zip(axes, sky.keys())):
        print(sky_key)
        data_model = data_plotting[f'{sky_key}_0']['data_model']
        data = data_plotting[f'{sky_key}_0']['data']

        intsky = data_model.integrate(data_model.rotation_and_shift(sky))
        integrated_sky.append(intsky)

        a0, a1, a2, a3 = axi
        a0.set_title(f'plaw {sky_key}')
        a1.set_title('high_res sky')
        a2.set_title('integrated sky')
        a3.set_title(f'data {sky_key}')

        ims = []
        ims.append(a0.imshow(plaw[ii], origin='lower', cmap='RdBu_r'))
        ims.append(a1.imshow(sky[sky_key], origin='lower', norm=LogNorm()))
        ims.append(a2.imshow(intsky, origin='lower', norm=LogNorm()))
        ims.append(a3.imshow(data, origin='lower', norm=LogNorm()))
        for ax, im in zip(axi, ims):
            plt.colorbar(im, ax=ax)

    first = integrated_sky[0]
    diffs = map(lambda y: first-y, integrated_sky[1:])
    for ii, (ax, diff) in enumerate(zip(axes[-1][1:], diffs)):
        im = ax.imshow(diff, origin='lower', cmap='RdBu_r')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'0 - {ii+1}')

    im = axes[-1][0].imshow(alpha, origin='lower')
    plt.colorbar(im, ax=axes[-1][0])

    plt.show()


pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

cfg_mini = ju.get_config('demos/jwst_mock_config.yaml')
minimization_config = cfg_mini['minimization']
kl_solver_kwargs = minimization_config.pop('kl_kwargs')
minimization_config['n_total_iterations'] = 12
# minimization_config['resume'] = True
minimization_config['n_samples'] = lambda it: 4 if it < 10 else 10

plot = build_plot(
    data_dict=data_plotting,
    sky_model_with_key=sky_model_with_keys,
    sky_model=sky_model,
    small_sky_model=small_sky_model,
    results_directory=RES_DIR,
    plaw=sky_model_new.plaw,
    plotting_config=dict(
        norm=LogNorm,
        sky_extent=None
    ))

print(f'Results: {RES_DIR}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,
    kl_kwargs=kl_solver_kwargs,
    callback=plot,
    odir=RES_DIR,
    **minimization_config)


field = jft.mean([sky_model_new.plaw(si) for si in samples])

fig, axes = plt.subplots(1, 3)
for ax, f in zip(axes, field):
    im = ax.imshow(f, origin='lower')
    plt.colorbar(im, ax=ax)
plt.show()
