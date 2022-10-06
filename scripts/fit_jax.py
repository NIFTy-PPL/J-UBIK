import sys
from collections import namedtuple
from functools import partial
import numpy as np
import yaml

from jax import jit, value_and_grad
from jax import random
from jax import numpy as jnp
from jax.config import config as jax_config
from jax.tree_util import tree_map

import nifty8 as ift
import nifty8.re as jft
import xubik0 as xu

jax_config.update("jax_enable_x64", True)
ift.set_nthreads(2)

cfg = xu.get_cfg("config.yaml")

npix_s = 1024  # number of spacial bins per axis
fov = 21.0
energy_bin = 0
position_space = ift.RGSpace([npix_s, npix_s], distances=[2.0 * fov / npix_s])

# Model P(s)
diffuse = ift.SimpleCorrelatedField(position_space, **cfg["priors_diffuse"])
pspec = diffuse.power_spectrum
diffuse = diffuse.exp()
points = ift.InverseGammaOperator(position_space, **cfg["points"])
points = points.ducktape("points")
signal =  diffuse# + points
signal = signal.real
signal_dt = signal.ducktape_left("full_signal")

# Likelihood P(d|s)
signal_fa = ift.FieldAdapter(signal_dt.target["full_signal"], "full_signal")
likelihood_list = []
# likelihood_list_nifty = []

for dataset in cfg["datasets"]:
    # Loop
    observation = np.load(dataset, allow_pickle=True).item()

    # PSF
    psf_arr = observation["psf_sim"].val[:, :, energy_bin]
    psf_arr = np.roll(psf_arr, -np.argmax(psf_arr))
    psf_field = ift.Field.from_raw(position_space, psf_arr)
    norm = ift.ScalingOperator(position_space, psf_field.integrate().val ** -1)
    psf = norm(psf_field)

    # Data
    data = observation["data"].val[:, :, energy_bin]
    data_field = ift.Field.from_raw(position_space, data)

    # Exp
    exp = observation["exposure"].val[:, :, energy_bin]
    exp_field = ift.Field.from_raw(position_space, exp)
    if dataset == cfg["datasets"][0]:
        norm_first_data = xu.get_norm(exp_field, data_field)
    normed_exp_field = ift.Field.from_raw(position_space, exp) * norm_first_data
    normed_exposure = ift.makeOp(normed_exp_field)

    # Mask
    mask = xu.get_mask_operator(normed_exp_field)

    # Likelihood
    conv = xu.convolve_field_operator(psf, signal_fa)
    signal_response = mask @ normed_exposure @ conv

    ############# JAX ########
    signal_response_jx, _ = ift.nifty2jax.convert(signal_response, float)
    ift.myassert(signal_response.jax_expr is signal_response_jx)

    masked_data = mask(data_field)
    likelihood = jft.Poissonian(masked_data.val) @ signal_response_jx
    likelihood_list.append(likelihood)
    # likelihood_nifty = ift.PoissonianEnergy(masked_data) @ signal_response
    # likelihood_list_nifty.append(likelihood_nifty)

likelihood_sum = likelihood_list[0]
for i in range(1, len(likelihood_list)):
    likelihood_sum = likelihood_sum + likelihood_list[i]


# likelihood_sum_nifty = likelihood_list_nifty[0]
# for i in range(1, len(likelihood_list_nifty)):
#     likelihood_sum_nifty = likelihood_sum_nifty + likelihood_list_nifty[i]

likelihood_sum = likelihood_sum @ signal_dt.jax_expr
lh = likelihood
ham = jft.StandardHamiltonian(likelihood=lh).jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
# likelihood_sum_nifty = likelihood_sum_nifty @ signal_dt
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name"),
)


# pos = ift.from_random(signal.domain).val
pt = ift.nifty2jax.shapewithdtype_from_domain(signal.domain, 'float')
pt = jft.Field(pt)
key = random.PRNGKey(42)
key, subkey = random.split(key)
pos = pos_init =  1e-2* jft.random_like(subkey, pt)


n_mgvi_iterations = 1
n_samples = 2
absdelta = 0.1
n_newton_iterations = 2

# Minimize the potential
key, *sk = random.split(key, 1 + n_mgvi_iterations)
for i, subkey in enumerate(sk):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    mg_samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
        # linear_sampling_name=None,
        # linear_sampling_kwargs={"absdelta": absdelta / 10.}
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=mg_samples),
            "hessp": partial(ham_metric, primals_samples=mg_samples),
            "absdelta": absdelta,
            "maxiter": n_newton_iterations
        }
    )
    pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {mg_samples.at(pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

# res1 = likelihood_sum(pos.val)
# res2 = likelihood_sum_nifty(pos)
# print(np.allclose(res1,res2.val))
transpose = xu.Transposer(signal.target)


def callback(samples):
    s = ift.extra.minisanity(
        masked_data,
        lambda x: ift.makeOp(1 / signal_response(signal_dt)(x)),
        signal_response(signal_dt),
        samples,
    )
    print(s)


global_it = cfg["global_it"]
n_samples = cfg["Nsamples"]
samples = ift.optimize_kl(
    likelihood_sum,
    global_it,
    n_samples,
    minimizer,
    ic_sampling,
    nl_sampling_minimizer,
    plottable_operators={
        "signal": transpose @ signal,
        "point_sources": transpose @ points,
        "diffuse": transpose @ diffuse,
        "power_spectrum": pspec,
    },
    output_directory="df_rec",
    initial_position=pos,
    comm=mpi.comm,
    inspect_callback=callback,
    overwrite=True,
    resume=True,
)
