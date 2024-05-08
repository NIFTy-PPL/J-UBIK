import yaml

import jubik0 as ju
from jubik0.jwst.mock_data import (
    setup, build_sky_model, build_evaluation_mask,
    build_mock_plot)
from jubik0.jwst.config_handler import config_transform, define_mock_output
from jubik0.jwst import build_data_model
from jubik0.jwst.likelihood import connect_likelihood_to_model

import nifty8.re as jft

from functools import reduce
import numpy as np
from astropy import units as u

from sys import exit
from jax import config, random
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')


cfg = yaml.load(
    open('demos/jwst_mock_config.yaml', 'r'), Loader=yaml.SafeLoader)
config_transform(cfg)
res_dir = define_mock_output(cfg)

# Draw random numbers
key = random.PRNGKey(87)
key, mock_key, noise_key, rec_key = random.split(key, 4)

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
likelihoods = []
for ii, (dkey, data_dict) in enumerate(data_set.items()):
    data, mask, std, data_grid = (
        data_dict['data'], data_dict['mask'], data_dict['std'],
        data_dict['grid'])

    data_model = build_data_model(
        reconstruction_grid=reco_grid,
        data_key=dkey,
        data_grid=data_grid,
        data_mask=mask,
        sky_model=jft.Model(
            jft.wrap_left(sky_model, internal_sky_key),
            domain=sky_model.domain),
        data_model_keyword=cfg['telescope']['rotation_model']['model'],
        subsample=cfg['telescope']['rotation_model']['subsample'],
        updating=False)
    data_dict['data_model'] = data_model

    world_extrema = data_grid.world_extrema
    to_be_subsampled_grid_wcs = data_grid.wcs
    index_grid_wcs = reco_grid.wcs
    subsample = 3

    exit()

    likelihood = ju.library.likelihood.build_gaussian_likelihood(
        data.reshape(-1), float(std))
    likelihood = likelihood.amend(data_model, domain=data_model.domain)
    likelihoods.append(likelihood)

likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(
    likelihood,
    jft.Model(jft.wrap_left(sky_model, internal_sky_key),
              domain=sky_model.domain)
)

evaluation_mask = build_evaluation_mask(reco_grid, data_set)
plot = build_mock_plot(
    data_set=data_set,
    comparison_sky=comp_sky,
    sky_model=sky_model,
    res_dir=res_dir,
    eval_mask=evaluation_mask,
)

pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

cfg = ju.get_config('demos/jwst_mock_config.yaml')
minimization_config = cfg['minimization']
kl_solver_kwargs = minimization_config.pop('kl_kwargs')
minimization_config['n_total_iterations'] = 25
# minimization_config['resume'] = True
minimization_config['n_samples'] = lambda it: 4 if it < 10 else 10

print(f'Results: {res_dir}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,
    kl_kwargs=kl_solver_kwargs,
    callback=plot,
    odir=res_dir,
    **minimization_config)
