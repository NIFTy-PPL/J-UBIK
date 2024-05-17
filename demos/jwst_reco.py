import matplotlib.pyplot as plt
import yaml
from functools import reduce

import nifty8.re as jft
from jax import config
from jax import random
import jax.numpy as jnp

import numpy as np

from matplotlib.colors import LogNorm

# import astropy
# import webbpsf
from astropy import units as u

import jubik0 as ju
from jubik0.library.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)
from jubik0.jwst.reconstruction_grid import Grid
from jubik0.jwst.jwst_data import JwstData
from jubik0.jwst.masking import get_mask_from_index_centers
from jubik0.jwst.config_handler import (define_location, get_shape, get_fov,
                                        get_rotation)
from jubik0.jwst.wcs import (subsample_grid_centers_in_index_grid)
from jubik0.jwst.jwst_model_builder import build_data_model
from jubik0.jwst.utils import build_sky_model
from jubik0.jwst.jwst_plotting import build_plot, plot_sky
from jubik0.jwst.filter_projector import FilterProjector


from sys import exit


# cfg.update('jax_enable_x64', True)
# cfg.update('jax_platform_name', 'cpu')


config_path = './demos/JWST_config.yaml'
cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
WORLD_LOCATION = define_location(cfg)
FOV = get_fov(cfg)
SHAPE = get_shape(cfg)
ROTATION = get_rotation(cfg)

res_dir = cfg['files']['res_dir']

# defining the reconstruction grid
reconstruction_grid = Grid(
    WORLD_LOCATION, SHAPE, (FOV.to(u.deg), FOV.to(u.deg)), rotation=ROTATION)
internal_sky_key = 'sky'

sky_model_new = ju.SkyModel(config_file_path=config_path)
small_sky_model = sky_model_new.create_sky_model()
sky_model = sky_model_new.full_diffuse


S_PADDING_RATIO = cfg['grid']['s_padding_ratio']
D_PADDING_RATIO = cfg['grid']['d_padding_ratio']
# small_sky_model, sky_model = build_sky_model(
#     reconstruction_grid.shape,
#     [d.to(u.arcsec).value for d in reconstruction_grid.distances],
#     cfg['priors']['diffuse']['spatial']['offset'],
#     cfg['priors']['diffuse']['spatial']['fluctuations'],
#     extend=S_PADDING_RATIO
# )
# sky_model_with_key = jft.Model(jft.wrap_left(sky_model, internal_sky_key),
#                                domain=sky_model.domain)

key = random.PRNGKey(87)
key, test_key, rec_key = random.split(key, 3)
x = jft.random_like(test_key, sky_model.domain)

# FIXME: This needs to provided somewhere else
SUBSAMPLE = cfg['telescope']['integration_model']['subsample']
MODEL_TYPE = 'linear'
DATA_DVOL = (0.13*u.arcsec**2).to(u.deg**2)


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

            subsample=SUBSAMPLE,

            rotation_and_shift_kwargs=dict(
                data_dvol=DATA_DVOL,
                data_wcs=jwst_data.wcs,
                data_model_type=MODEL_TYPE,
            ),

            psf_kwargs=dict(
                camera=jwst_data.camera,
                filter=jwst_data.filter,
                center_pixel=jwst_data.wcs.index_from_wl(
                    reconstruction_grid.center)[0],
                webbpsf_path=cfg['telescope']['web_psf']['webpsf_path'],
                psf_library_path=cfg['telescope']['web_psf']['psf_library'],
                fov_pixels=32,
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

    fig, axes = plt.subplots(len(sky), 3)
    ims = []
    for axi, sky_key in zip(axes, sky.keys()):
        print(sky_key)
        data_model = data_plotting[f'{sky_key}_0']['data_model']
        data = data_plotting[f'{sky_key}_0']['data']

        ax, az, aa = axi
        ax.set_title('high_res sky')
        az.set_title('integrated sky')
        aa.set_title(f'data {sky_key}')
        ims.append(ax.imshow(sky[sky_key], origin='lower', norm=LogNorm()))
        ims.append(az.imshow(
            data_model.integrate(data_model.rotation_and_shift(sky)),
            origin='lower', norm=LogNorm()))
        ims.append(aa.imshow(data, origin='lower', norm=LogNorm()))
        for ax, im in zip(axi, ims):
            plt.colorbar(im, ax=ax)
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
    results_directory=res_dir,
    plotting_config=dict(
        norm=LogNorm,
        sky_extent=None
    ))

print(f'Results: {res_dir}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,
    kl_kwargs=kl_solver_kwargs,
    callback=plot,
    odir=res_dir,
    **minimization_config)

sky = jft.mean([sky_model_with_keys(si) for si in samples])

rs = data_model.rotation_and_shift(sky)
p = data_model.psf(rs)
i = data_model.integrate(p)

fig, axes = plt.subplots(1, 4)
ax, ay, az, aa = axes
ax.imshow(sky['sky'], origin='lower', norm=LogNorm())
ay.imshow(rs, origin='lower', norm=LogNorm())
az.imshow(p, origin='lower', norm=LogNorm())
aa.imshow(i, origin='lower', norm=LogNorm())
aa.contour(mask)
plt.show()
