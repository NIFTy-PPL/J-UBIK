import nifty8.re as jft

import jubik0 as ju
from jubik0.jwst.mock_data import (
    setup, build_evaluation_mask, build_mock_plot)
from jubik0.jwst.utils import build_sky_model
from jubik0.jwst.config_handler import config_transform, define_mock_output
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.library.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)

from functools import reduce
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
sky_model, sky_model_full = build_sky_model(
    reco_grid.shape,
    [d.to(u.arcsec).value for d in reco_grid.distances],
    cfg['sky_model']['offset'],
    cfg['sky_model']['fluctuations'],
    extend_factor=cfg['sky_model']['extend_factor'],
)
if cfg['sky_model'].get('plot_sky_model', False):
    from jubik0.jwst.mock_data.mock_plotting import sky_model_check
    key, check_key = random.split(key)
    sky_model_check(check_key, sky_model, comp_sky)


internal_sky_key = 'sky'
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
            data_model_type=cfg['telescope']['rotation_and_shift']['model'],
            kwargs_sparse=dict(extend_factor=1,  # cfg['sky_model']['extend_factor'],
                               to_bottom_left=True),
        ),
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
likelihood = connect_likelihood_to_model(
    likelihood,
    sky_model_with_key
)

exit()

evaluation_mask = build_evaluation_mask(reco_grid, data_set)
plot = build_mock_plot(
    data_set=data_set,
    comparison_sky=comp_sky,
    internal_sky_key=internal_sky_key,
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
