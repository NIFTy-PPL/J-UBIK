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

from charm_lensing import minimization_parser
# from charm_lensing import config_parser, build_lens_system

from sys import exit

if False:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])

config_path = './demos/JWST_config.yaml'
cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
RES_DIR = cfg['files']['res_dir']
D_PADDING_RATIO = cfg['grid']['d_padding_ratio']
# FIXME: This needs to provided somewhere else
DATA_DVOL = (0.13*u.arcsec**2).to(u.deg**2)
PSF_PIXELS = 32

reconstruction_grid = build_reconstruction_grid_from_config(cfg)

sky_model_new = ju.SkyModel(config_file_path=config_path)
small_sky_model = sky_model_new.create_sky_model(fov=cfg['grid']['fov'])
sky_model = sky_model_new.full_diffuse


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
kk = 0
for fltname, flt in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        jwst_data = JwstData(filepath)

        data_key = f'{fltname}_{ii}'

        if kk == 0:
            correction_model = None
        else:
            from jubik0.jwst.rotation_and_shift.coordinates_correction import build_coordinates_correction_model
            header = jwst_data.wcs._wcs.to_fits()[0]
            rpix = (header['CRPIX1'],),  (header['CRPIX2'],)
            rpix = jwst_data.wcs.wl_from_index(rpix)
            rpix = reconstruction_grid.wcs.index_from_wl(rpix)[0]

            correction_model = build_coordinates_correction_model(
                data_key + '_correction',
                dict(shift=(0, 1.0e-1),
                     # rotation=(0, (1.0e-1*u.deg).to(u.rad).value)),
                     rotation=(0, (1.0e-1*u.deg).to(u.rad).value)),
                pix_distance=[
                    rd.to(u.arcsec).value for rd in reconstruction_grid.distances],
                rotation_center=rpix)

        kk += 1

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
                shift_and_rotation_correction=correction_model,
            ),

            psf_kwargs=dict(
                camera=jwst_data.camera,
                filter=jwst_data.filter,
                center_pixel=jwst_data.wcs.index_from_wl(
                    reconstruction_grid.center)[0],
                webbpsf_path=cfg['telescope']['psf']['webbpsf_path'],
                psf_library_path=cfg['telescope']['psf']['psf_library'],
                fov_pixels=PSF_PIXELS,
            ),

            data_mask=mask,

            world_extrema=reconstruction_grid.world_extrema(D_PADDING_RATIO)
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

for ii in range(0):
    key, test_key = random.split(random.PRNGKey(42+ii), 2)
    x = jft.random_like(test_key, likelihood.domain).tree
    sky = sky_model_with_keys(x)
    # plot_sky(sky, data_plotting)

    plaw = sky_model_new.plaw(x)
    alpha = sky_model_new.alpha_cf(x)

    # k = 'f356w'
    # a, b = (data_plotting[k + '_0']['data_model'],
    #         data_plotting[k+'_1']['data_model'])
    # val = sky | x
    # sa, sb = a.rotation_and_shift(val), b.rotation_and_shift(val)

    # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True,)
    # ax, ay, az = axes
    # ax.imshow(sa, origin='lower')
    # ay.imshow(sb, origin='lower')
    # az.imshow(sa-sb[:, :-3], origin='lower')
    # plt.show()

    # exit()

    fig, axes = plt.subplots(len(sky)+1, 4)
    integrated_sky = []
    for ii, (axi, sky_key) in enumerate(zip(axes, sky.keys())):
        print(sky_key)
        data_model = data_plotting[f'{sky_key}_0']['data_model']
        correction_model = data_plotting[f'{sky_key}_0']['correction_model']
        data = data_plotting[f'{sky_key}_0']['data']

        val = sky | x
        intsky = data_model.integrate(data_model.rotation_and_shift(val))
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
    axes[-1][0].set_title('alpha')
    plt.colorbar(im, ax=axes[-1][0])

    plt.show()


cfg_mini = ju.get_config('demos/JWST_config.yaml')["minimization"]
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
