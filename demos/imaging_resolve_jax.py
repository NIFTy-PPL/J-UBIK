import jax
from jax import numpy as jnp

import nifty8.re as jft
import resolve as rve_old
import jubik0.instruments.resolve as rve
import jubik0.instruments.resolve.re as jrve

from jubik0.instruments.resolve.re.parse.response import (
    SkyDomain, Ducc0Settings, FinufftSettings)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import configparser
from jax import random

from jax import config, devices
config.update('jax_default_device', devices('cpu')[0])
jax.config.update("jax_enable_x64", True)


# choose between ducc0 and finufft backend
response = 'ducc0'
# response = "finufft"
backend_settings = dict(
    ducc0=Ducc0Settings(
        epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False),
    finufft=FinufftSettings(epsilon=1e-9),
)[response]

seed = 42
key = random.PRNGKey(seed)


obs = rve.Observation.load("CYG-ALL-2052-2MHZ_RESOLVE_float64.npz")
obs = obs.restrict_to_stokesi()
obs = obs.average_stokesi()
# scale weights, as they are wrong for this specific dataset
obs._weight = 0.1 * obs._weight
cfg = configparser.ConfigParser()
cfg.read("cygnusa_2ghz.cfg")

sky, additional = jrve.sky_model(cfg["sky"])


sky_sp = rve_old.sky_model._spatial_dom(cfg["sky"])
sky_dom = rve_old.default_sky_domain(sdom=sky_sp)


sky_domain = SkyDomain(npix_x=sky_sp.shape[0],
                       npix_y=sky_sp.shape[1],
                       pixsize_x=sky_sp.distances[0],
                       pixsize_y=sky_sp.distances[1],
                       polarization_labels=['I'],
                       times=[0.],
                       frequencies=[0.])
R_new = jrve.InterferometryResponse(
    obs, sky_domain, backend_settings=backend_settings)


def signal_response(x): return R_new(sky(x))


nll = jft.Gaussian(obs.vis_val, obs.weight_val).amend(signal_response)


def callback(samples, opt_state):
    post_sr_mean = jft.mean(tuple(sky(s) for s in samples))
    plt.imshow(post_sr_mean[0, 0, 0, :, :].T, origin="lower", norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"niftyre_it_{opt_state.nit}_response_{response}.png")
    plt.close()


n_vi_iterations = 20
delta = 1e-8
absdelta = delta * jnp.prod(jnp.array(sky_sp.shape))
n_samples = 2


def sample_mode_update(i):
    return "linear_resample"


def draw_linear_kwargs(i):
    return dict(cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=250))


def kl_kwargs(i):
    return dict(
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
    odir=f"results_nifty_re_response_{response}",
    resume=False,
    callback=callback,
)
