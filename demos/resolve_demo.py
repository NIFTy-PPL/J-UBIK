import configparser
from os.path import join

import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import numpy as jnp
from jax import random
from matplotlib.colors import LogNorm
from astropy import units as u

import jubik0 as ju
import jubik0.instruments.resolve as rve
from jubik0.parse.grid import GridModel
from jubik0.sky_model.resolve_sky import sky_model

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

cfg = configparser.ConfigParser()
cfg.read("./demos/configs/cygnusa_2ghz.cfg")

# choose between ducc0 and finufft backend
response = "ducc0"
# response = "finufft"
backend_settings = dict(
    ducc0=rve.parse.Ducc0Settings(
        epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False
    ),
    finufft=rve.parse.FinufftSettings(epsilon=1e-9),
)[response]

# # NOTE : The observation can also be loaded and modified via the config file.
# backend_settings = config_parser_to_response_settings(cfg['data'])

obs = rve.Observation.load("./data/resolve_test/CYG-ALL-2052-2MHZ_RESOLVE_float64.npz")
obs = rve.data.restrict_to_stokesi(obs)
obs = rve.data.average_stokesi(obs)

# scale weights, as they are wrong for this specific dataset
obs._weight = 0.1 * obs._weight
obs = rve.data.select_random_visibility_subset(obs, 0.01)

# # NOTE : The observation can also be loaded and modified via the config file.
# from jubik0.instruments.resolve.data import load_and_modify_data_from_objects
# obs = list(
#     load_and_modify_data_from_objects(
#         [-np.inf, np.inf],
#         DataLoading.from_config_parser(cfg['data']),
#         ObservationModify.from_config_parser(cfg['data'])))[0]

sky, additional = sky_model(cfg["sky"])


# gm = GridModel.from_config_parser(cfg["sky"])
# gm.spatial_model.wcs_model.center = obs.direction.to_sky_coord()
# grid = ju.Grid.from_grid_model(gm)

grid = ju.Grid.from_shape_and_fov(
    shape=(int(cfg["sky"]["space npix x"]), int(cfg["sky"]["space npix y"])),
    fov=u.Quantity(
        (u.Quantity(cfg["sky"]["space fov x"]), u.Quantity(cfg["sky"]["space fov y"]))
    ),
)


R_new = rve.interferometry_response(obs, grid, backend_settings=backend_settings)


def signal_response(x):
    return R_new(sky(x))


nll = jft.Gaussian(obs.vis_val, obs.weight_val).amend(signal_response)

output_dir = "imaging_resolve"


def callback(samples, opt_state):
    post_sr_mean = jft.mean(tuple(sky(s) for s in samples))
    plt.imshow(post_sr_mean[0, 0, 0, :, :].T, origin="lower", norm=LogNorm())
    plt.colorbar()
    plt.savefig(join(output_dir, f"iteration_{opt_state.nit}.png"))
    plt.close()


n_vi_iterations = 20
delta = 1e-8
absdelta = delta * jnp.prod(jnp.array(grid.spatial.shape))
n_samples = 2

sample_mode_update = "linear_resample"
draw_linear_kwargs = dict(
    cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=250)
)
kl_kwargs = dict(
    minimize_kwargs=dict(
        name="M", absdelta=absdelta, cg_kwargs=dict(name="MCG"), maxiter=20
    )
)

key, subkey = random.split(key)
pos_init = jft.Vector(jft.random_like(subkey, sky.domain))
samples, state = jft.optimize_kl(
    nll,
    pos_init,
    n_total_iterations=n_vi_iterations,
    n_samples=n_samples,
    key=key,
    draw_linear_kwargs=draw_linear_kwargs,
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=10,
        )
    ),
    kl_kwargs=kl_kwargs,
    sample_mode=sample_mode_update,
    odir=output_dir,
    resume=False,
    callback=callback,
)
