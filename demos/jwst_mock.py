from sys import exit
import nifty8.re as jft

import numpy as np

from jubik0.jwst.mock_data import setup, build_sky_model, build_data_model
from jubik0.jwst.mock_plotting import (build_evaluation_mask, build_plot)
from jubik0.jwst.likelihood import connect_likelihood_to_model

from functools import reduce
import jubik0 as ju

from astropy.coordinates import SkyCoord
from astropy import units as u

from jax import config, random
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')


# Draw random numbers
key = random.PRNGKey(87)
key, mock_key, noise_key, rec_key = random.split(key, 4)

# Sky setup
MOCK_SHAPE = 1536
ROTA_SHAPE = 768
RECO_SHAPE = 256
DATA_SHAPE = 48
SHIFTS = [(10, 0), (0, 0)]
REPORTED_SHIFTS = [(10, 0), (0, 0)]
ROTATIONS = [2, 23]
REPORTED_ROTATIONS = [2, 23]
SKY_DICT = dict(
    offset=dict(offset_mean=0.1, offset_std=[0.1, 0.05]),
    fluctuations=dict(fluctuations=[0.3, 0.03], loglogavgslope=[-3., 1.],
                      flexibility=[0.8, 0.1], asperity=[0.2, 0.1])
)
PLOT_SETUP = False

# Reconstruction setup
SUBSAMPLE = 2
NOISE_SCALE = 0.01
MODEL = 'linear'  # linear, nufft, sparse
PLOT_SKYMODEL = False

# Results
rot_string = 'r' + '_'.join([f'{r}' for r in ROTATIONS])
met_string = MODEL if MODEL == 'sparse' else MODEL + f'{SUBSAMPLE}'
sh_string = 's' + '_'.join([f'{np.hypot(*r)}' for r in SHIFTS])
res_dir = f'results/mock_data/{RECO_SHAPE}_{met_string}/{rot_string}_{sh_string}/'


comp_sky, reco_grid, data_set = setup(
    mock_key,
    rotation=ROTATIONS,
    repo_rotation=REPORTED_ROTATIONS,
    shift=SHIFTS,
    repo_shift=REPORTED_SHIFTS,
    reco_shape=RECO_SHAPE,
    mock_shape=MOCK_SHAPE,
    rota_shape=ROTA_SHAPE,
    data_shape=DATA_SHAPE,
    plot=PLOT_SETUP,
)

internal_sky_key = 'SKY_KEY'
sky_model = build_sky_model(
    internal_sky_key,
    reco_grid.shape,
    [d.to(u.arcsec).value for d in reco_grid.distances])
if PLOT_SKYMODEL:
    from jwst_handling.mock_data import sky_model_check
    key, check_key = random.split(key)
    sky_model_check(check_key, sky_model, comp_sky)


likelihood_dicts = {}
for ii, (dkey, data_dict) in enumerate(data_set.items()):
    data, data_grid = data_dict['data'], data_dict['grid']

    # Create noise
    std = data.mean() * NOISE_SCALE
    d = data + random.normal(noise_key, data.shape, dtype=data.dtype) * std
    mask = np.full(data.shape, True)

    data_model = build_data_model(
        reco_grid=reco_grid,
        data_key=dkey,
        data_grid=data_grid,
        data_mask=mask,
        sky_key=internal_sky_key,
        sky_model=sky_model,
        data_model_keyword=MODEL,
        subsample=SUBSAMPLE,
        updating=False)

    likelihood = ju.library.likelihood.build_gaussian_likelihood(
        d.reshape(-1), float(std))
    likelihood = likelihood.amend(data_model, domain=data_model.domain)
    likelihood_dicts[dkey] = dict(
        data=d, std=std, mask=mask, data_model=data_model, likelihood=likelihood)

likelihood = reduce(
    lambda x, y: x+y,
    [ll['likelihood'] for ll in likelihood_dicts.values()]
)
likelihood = connect_likelihood_to_model(likelihood, sky_model)

evaluation_mask = build_evaluation_mask(reco_grid, data_set)

plot = build_plot(
    likelihood_dicts=likelihood_dicts,
    comparison_sky=comp_sky,
    sky_key=internal_sky_key,
    sky_model=sky_model,
    res_dir=res_dir,
    eval_mask=evaluation_mask,
)

pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

cfg = ju.get_config('./JWST_config.yaml')
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
