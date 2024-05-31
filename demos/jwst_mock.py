from jubik0.jwst.integration_model.sum_integration import build_sum_integration
import matplotlib.pyplot as plt
import nifty8.re as jft

import jubik0 as ju
from jubik0.jwst.mock_data import (
    mock_setup, build_evaluation_mask, build_mock_plot)
from jubik0.jwst import ConnectModels
from jubik0.jwst.utils import build_sky_model
from jubik0.jwst.config_handler import config_transform, define_mock_output
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.library.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)
from jubik0.jwst.rotation_and_shift.shift_model import build_shift_model

from functools import reduce
from sys import exit

import yaml
from astropy import units as u
from jax import config, random


cfg = yaml.load(
    open('demos/jwst_mock_config.yaml', 'r'), Loader=yaml.SafeLoader)
config_transform(cfg)
res_dir = define_mock_output(cfg)

if cfg['mock_setup']['rotation'] == 'nufft':
    print("Using cpu")
    config.update('jax_enable_x64', True)
    config.update('jax_platform_name', 'cpu')

# Draw random numbers
key = random.PRNGKey(87)
key, mock_key, noise_key, rec_key, test_key = random.split(key, 5)

comp_sky, reco_grid, data_set = mock_setup(
    mock_key, noise_key, **cfg['mock_setup'])
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

    if ii == 1:
        shift_model = None
    else:
        subsample = cfg['telescope']['rotation_and_shift']['subsample']
        dist = [rd.to(u.arcsec).value for rd in reco_grid.distances]
        shift_model = build_shift_model(dkey + '_shift_cor', (0, 1.0), dist)

    data_grid = data['grid']
    data_model = build_data_model(
        sky_domain=sky_model_with_key.target,
        reconstruction_grid=reco_grid,

        subsample=cfg['telescope']['rotation_and_shift']['subsample'],

        rotation_and_shift_kwargs=dict(
            data_dvol=data_grid.dvol,
            data_wcs=data_grid.wcs,
            data_model_type=cfg['telescope']['rotation_and_shift']['model'],
            kwargs_sparse=dict(extend_factor=1, to_bottom_left=True),
            shift_and_rotation_correction_domain=shift_model.target if shift_model is not None else None,
            kwargs_linear=dict(order=1, sky_as_brightness=False, mode='wrap'),
        ),

        psf_kwargs=dict(),

        data_mask=data['mask'],

        world_extrema=data_grid.world_extrema())

    data['data_model'] = data_model
    data['correction_model'] = shift_model

    likelihood = build_gaussian_likelihood(
        data['data'].reshape(-1), float(data['std']))
    likelihood = likelihood.amend(
        data_model, domain=jft.Vector(data_model.domain))
    likelihoods.append(likelihood)


models = [sky_model_with_key] + [
    dm['correction_model'] for dm in data_set.values() if isinstance(dm['correction_model'], jft.Model)]
model = ConnectModels(models)

likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(likelihood, model)

# integration = build_sum_integration(sky_model.target.shape, 8)
# x = {'sky': comp_sky}
# data_no_noise = data['data_no_noise']
# mod_sky = data_model(x).reshape(data_no_noise.shape)

# fig, axes = plt.subplots(1, 3)
# ax, ay, az = axes
# ax.imshow(data_no_noise)
# ay.imshow(mod_sky)
# im = az.imshow((data_no_noise-mod_sky)/data_no_noise)
# plt.colorbar(im, ax=az)
# plt.show()

# exit()


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
# minimization_config['resume'] = True

print(f'Results: {res_dir}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,
    kl_kwargs=kl_solver_kwargs,
    callback=plot,
    odir=res_dir,
    **minimization_config)

# data_models = [data_model(sky_model_with_key(si)) for si in samples]

# im = plt.imshow((data['data'] - mod_mean), origin='lower')
# plt.colorbar(im)
# plt.show()

# sky = np.zeros(mask.shape)
# for si in samples:
#     sky[mask] += data_model(sky_model_with_key(si))
# sky = sky / len(samples)

# fig, axes = plt.subplots(3, 3, sharey=True, sharex=True)

# mask = data['mask']

# for si, ax in zip(samples, axes.flatten()):
#     sky = np.zeros(mask.shape)
#     sky[mask] = data_model(sky_model_with_key(si))
#     im = ax.imshow(sky, origin='lower')
#     plt.colorbar(im, ax=ax)

# plt.show()
