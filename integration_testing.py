import nifty8.re as jft

import jubik0 as ju
from jubik0.jwst.mock_data import mock_setup
from jubik0.jwst.utils import build_sky_model
from jubik0.jwst.integration_model import build_integration
from jubik0.library.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)
from jubik0.jwst.jwst_data_model import build_data_model

from functools import reduce

from numpy import allclose

import yaml
from astropy import units as u
from jax import config, random
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')


cfg = yaml.load(
    open('demos/jwst_mock_config.yaml', 'r'), Loader=yaml.SafeLoader)
cfg['mock_setup']['mock_shape'] = cfg['mock_setup']['mock_shape'] // 8
cfg['mock_setup']['rota_shape'] = cfg['mock_setup']['rota_shape'] // 8
cfg['mock_setup']['reco_shape'] = cfg['mock_setup']['reco_shape'] // 8
cfg['mock_setup']['data_shape'] = cfg['mock_setup']['data_shape'] // 8
cfg['mock_setup']['mock_distance'] = cfg['mock_setup']['mock_distance'] * 8
cfg['mock_setup']['shifts'] = [cfg['mock_setup']['shifts'][0]]
cfg['mock_setup']['rotations'] = [cfg['mock_setup']['rotations'][0]]

# Draw random numbers
key = random.PRNGKey(87)
key, mock_key, noise_key, test_key = random.split(key, 4)

comp_sky, reco_grid, data_set = mock_setup(
    mock_key, noise_key, **cfg['mock_setup'])
sky_model, sky_model_full = build_sky_model(
    reco_grid.shape,
    [d.to(u.arcsec).value for d in reco_grid.distances],
    cfg['sky_model']['offset'],
    cfg['sky_model']['fluctuations'],
)

internal_sky_key = 'sky'

MODEL_TYPE = cfg['telescope']['rotation_and_shift']['model']

sky_model_with_key = jft.Model(jft.wrap_left(sky_model, internal_sky_key),
                               domain=sky_model.domain)

likelihoods = []
for ii, (dkey, data) in enumerate(data_set.items()):

    data_grid = data['grid']
    data_model = build_data_model(
        sky_domain=sky_model_with_key.target,
        reconstruction_grid=reco_grid,
        subsample=cfg['telescope']['rotation_and_shift']['subsample'],
        rotation_and_shift_kwargs=dict(
            data_dvol=data_grid.dvol,
            data_wcs=data_grid.wcs,
            data_model_type=MODEL_TYPE),
        psf_kwargs=dict(),
        data_mask=data['mask'],
        world_extrema=data_grid.world_extrema())

    data['data_model'] = data_model

    likelihood = build_gaussian_likelihood(
        data['data'].reshape(-1), float(data['std']))
    likelihood = likelihood.amend(
        data_model, domain=jft.Vector(data_model.domain))
    likelihoods.append(likelihood)

likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood_new = connect_likelihood_to_model(
    likelihood,
    sky_model_with_key
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
        sky_model=sky_model_with_key,
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
    likelihood, sky_model_with_key)

key = random.PRNGKey(87)
key, test_key = random.split(key, 2)
x = jft.random_like(test_key, sky_model.domain)
assert allclose(likelihood_new(x), likelihood_old(x))
