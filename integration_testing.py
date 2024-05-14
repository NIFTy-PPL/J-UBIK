import numpy as np
from jubik0.jwst.wcs import subsample_grid_centers_in_index_grid
import matplotlib.pyplot as plt
from jubik0.jwst.rotation_and_shift import build_rotation_and_shift_model

import nifty8.re as jft

import jubik0 as ju
from jubik0.jwst.mock_data import setup
from jubik0.jwst.utils import build_sky_model
from jubik0.jwst.config_handler import config_transform, define_mock_output
from jubik0.jwst.integration_model import build_integration
from jubik0.jwst.likelihood import connect_likelihood_to_model
from jubik0.jwst.jwst_model_builder import build_jwst_model

from copy import deepcopy
from functools import reduce

from numpy import allclose


from sys import exit
import yaml
from astropy import units as u
from jax import config, random
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')


cfg = yaml.load(
    open('demos/jwst_mock_config.yaml', 'r'), Loader=yaml.SafeLoader)
config_transform(cfg)
res_dir = define_mock_output(cfg)

# Draw random numbers
key = random.PRNGKey(87)
key, mock_key, noise_key, rec_key, test_key = random.split(key, 5)

comp_sky, reco_grid, data_set = setup(mock_key, noise_key, **cfg['mock_setup'])
sky_model = build_sky_model(
    reco_grid.shape,
    [d.to(u.arcsec).value for d in reco_grid.distances],
    cfg['sky_model']['offset'],
    cfg['sky_model']['fluctuations'],
)
if cfg['sky_model'].get('plot_sky_model', False):
    from jubik0.jwst.mock_data.mock_plotting import sky_model_check
    key, check_key = random.split(key)
    sky_model_check(check_key, sky_model, comp_sky)


internal_sky_key = 'sky'

data_set_new = deepcopy(data_set)
for key, val in data_set_new.items():
    val['model_type'] = 'linear'
    val['subsample'] = cfg['telescope']['rotation_and_shift']['subsample']


# smodel = jft.Model(jft.wrap_left(sky_model, internal_sky_key),
#                    domain=sky_model.domain)
# dkey, data_dict = next(iter(data_set_new.items()))
# rotation_and_shift = build_rotation_and_shift_model(
#     smodel.target,
#     world_extrema_key='from_data',
#     reconstruction_grid=reco_grid,
#     data_key=dkey,
#     data_grid=data_dict['grid'],
#     **data_dict)

# test_val = jft.random_like(test_key, sky_model.domain)
# test_sky = smodel(test_val)

# sky = rotation_and_shift(test_sky)

# fig, axes = plt.subplots(2, 2)
# axes = axes.flatten()
# for ax, s in zip(axes, sky):
#     ax.imshow(s)
# plt.show()


# reconstruction_grid = reco_grid
# data_grid = data_dict['grid']
# subsample = data_dict['subsample']
# world_extrema = data_grid.world_extrema

# to_be_subsampled_grid_wcs = data_grid.wcs
# index_grid_wcs = reco_grid.wcs
# ssg_pixcenter_indices = to_be_subsampled_grid_wcs.index_grid_from_wl_extrema(
#     world_extrema)

# ps = np.arange(0.5/subsample, 1, 1/subsample) - 0.5
# ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)
# subsample_centers = ms[:, :, None, None] + ssg_pixcenter_indices

# subsample_centers = subsample_grid_centers_in_index_grid(
#     world_extrema,
#     data_grid.wcs,
#     reconstruction_grid.wcs,
#     subsample)


# def subsample_stack(array, subsample):
#     new_array = np.zeros(
#         (2, subsample*array.shape[-2], subsample*array.shape[-1]))

#     for ii in range(array.shape[0]):
#         xx = ii % 2
#         yy = ii // 2
#         new_array[:, xx::subsample, yy::subsample] = array[ii]

#     return new_array
# exit()


likelihood = build_jwst_model(
    sky_model=jft.Model(jft.wrap_left(sky_model, internal_sky_key),
                        domain=sky_model.domain),
    reconstruction_grid=reco_grid,
    data_set=data_set_new,
    world_extrema_key='from_data')
likelihood_new = connect_likelihood_to_model(
    likelihood,
    jft.Model(jft.wrap_left(sky_model, internal_sky_key),
              domain=sky_model.domain)
)

likelihoods = []
for ii, (dkey, data_dict) in enumerate(data_set.items()):
    data, mask, std, data_grid = (
        data_dict['data'], data_dict['mask'], data_dict['std'],
        data_dict['grid'])

    data_model = build_integration(
        reconstruction_grid=reco_grid,
        data_grid=data_grid,
        data_mask=mask,
        sky_model=jft.Model(
            jft.wrap_left(sky_model, internal_sky_key),
            domain=sky_model.domain),
        data_model_keyword=cfg['telescope']['rotation_and_shift']['model'],
        subsample=cfg['telescope']['rotation_and_shift']['subsample'],
        updating=False)
    data_dict['data_model'] = data_model

    likelihood = ju.library.likelihood.build_gaussian_likelihood(
        data.reshape(-1), float(std))
    likelihood = likelihood.amend(data_model, domain=data_model.domain)
    likelihoods.append(likelihood)

likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood_old = connect_likelihood_to_model(
    likelihood,
    jft.Model(jft.wrap_left(sky_model, internal_sky_key),
              domain=sky_model.domain)
)

test_val = jft.random_like(test_key, sky_model.domain)
assert allclose(likelihood_new(test_val), likelihood_old(test_val))
