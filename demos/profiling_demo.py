# %% [markdown]
# # Demo: Profiling compiled models and their sub-models
#
# This demo shows how to use `jubik.profiling` to measure, per (sub-)model:
#
# - compile time (tracing + XLA compilation),
# - steady-state runtime (min-of-N, blocking on every call),
# - gradient (VJP) compile time and runtime — inference is
#   gradient-dominated, so this often matters more than the forward pass,
# - XLA compiler estimates: flops, bytes accessed, temp/argument/output
#   buffer sizes,
# - measured peak device memory (GPU/TPU only).
#
# The one thing to keep in mind: `jax.jit` fuses the whole model into a
# single XLA executable, so there is no exact per-sub-model breakdown of
# the fused program. `profile_tree` therefore measures each sub-model
# jit-compiled *in isolation* and additionally measures the fused root —
# the ratio between the two (the "fusion gap") tells you how much work
# XLA saves by fusing across sub-model boundaries. The numbers are meant
# for *relative* comparison between sub-models and for tracking
# regressions, not as absolute ground truth.
#
# Run with:
#
#     python profiling_demo.py
#
# It is CPU-friendly and needs no data downloads. Note that on the CPU
# backend `peak_bytes` is unavailable and flops/bytes estimates may be
# missing on older jax versions; run on GPU for the full picture.

# %%
import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import nifty.re as jft

import jubik as ju

# %% [markdown]
# ## Build a small model tree
#
# A miniature version of the usual jubik setup: a correlated-field sky,
# an exponential link, and a "response" that downsamples and masks. Each
# stage is a `jft.Model`, composed exactly as in the instrument modules.

# %%
cfm = jft.CorrelatedFieldMaker('sky_')
cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1e-1, 1e-2))
cfm.add_fluctuations((256, 256), (1.0, 1.0), fluctuations=(1., .5),
                     loglogavgslope=(-3., .5), flexibility=None, asperity=None)
diffuse = cfm.finalize()

sky = jft.Model(lambda x: jnp.exp(diffuse(x)), domain=diffuse.domain)

mask = jnp.ones((128, 128), dtype=bool).at[32:96, 32:96].set(False)
response = jft.Model(lambda x: sky(x)[::2, ::2][mask], domain=sky.domain)

# %% [markdown]
# ## Profile a single model
#
# `profile_model` works on anything with a `.domain` (a `jft.Model` or a
# `jft.Likelihood`); the example input is drawn automatically from it.
# With `grad=True` the VJP is compiled and timed as well.

# %%
row = ju.profile_model(sky, name='sky', grad=True, n=20)
print(f'{row.name}: {row.n_params} params, '
      f'compile {row.compile_s:.3f}s, runtime {row.runtime_s * 1e3:.2f}ms, '
      f'grad runtime {row.grad_runtime_s * 1e3:.2f}ms')

# %% [markdown]
# ## Profile a tree of named sub-models against the fused whole
#
# Pass the sub-models under display names plus the full composition as
# `root`. The last line of the table reports the fusion gap: with a value
# well above 1, the isolated per-sub-model numbers overcount what the
# sub-models cost inside the fused executable.

# %%
report = ju.profile_tree(
    {'diffuse (cf)': diffuse, 'sky (exp)': sky, 'response': response},
    root=response, grad=True, n=20, verbose=False)
print(report)

# %% [markdown]
# The report can be persisted next to other run outputs, e.g. to compare
# against a later commit or a different grid size.

# %%
report.to_json('profile.json')

# %% [markdown]
# ## Plain callables (no `.domain`)
#
# Parts of jubik are plain closures without shape metadata (the eROSITA
# response dict, `unit_conversion`, masks). Those need an explicit
# example input.

# %%
def unit_conversion(x):
    return 4.2 * x

row = ju.profile_model(unit_conversion, x=jnp.ones((128, 128)),
                       name='unit_conversion', n=20)
print(f'{row.name}: runtime {row.runtime_s * 1e6:.0f}us')

# %% [markdown]
# ## Monitoring an `optimize_kl` run
#
# `ProfilingCallback` drops into the callback chain of `jft.optimize_kl`
# and records, per iteration, the wall time since the previous iteration,
# device memory in use, its peak, and the number of live device arrays —
# one JSON line each, plus a log line. Memory growth over iterations
# (e.g. compilation-cache bloat on sample-mode switches) becomes visible
# instead of guessed; pair it with `ju.clear_jax_compilation_cache` to
# verify the clearing actually helps.

# %%
key = jax.random.PRNGKey(42)
key, subkey, datakey = jax.random.split(key, 3)

synth_truth = response(response.init(datakey))
data = synth_truth + 0.1 * jax.random.normal(datakey, synth_truth.shape)
likelihood = jft.Gaussian(data, noise_std_inv=lambda x: x / 0.1).amend(response)

profiling_callback = ju.ProfilingCallback(path='profile_iterations.jsonl')

samples, state = jft.optimize_kl(
    likelihood,
    0.1 * jft.Vector(jft.random_like(subkey, likelihood.domain)),
    key=key,
    n_total_iterations=3,
    n_samples=2,
    draw_linear_kwargs=dict(cg_kwargs=dict(maxiter=20)),
    kl_kwargs=dict(minimize_kwargs=dict(maxiter=5)),
    callback=profiling_callback,
    odir=None,
)

# %% [markdown]
# `profile_iterations.jsonl` now holds one record per iteration:
#
#     {"nit": 2, "wall_s": 1.8, "bytes_in_use": ..., "peak_bytes_in_use": ..., "n_live_arrays": ...}
#
# ## charm_lensing systems
#
# For a charm_lensing `CompiledSystem`, every addressable sub-model can be
# enumerated automatically and fed straight into `profile_tree`:
#
#     from charm_lensing.lens_system import build_lens_system
#     from charm_lensing.algebra.lens import sky
#
#     system = build_lens_system(cfg)
#     named = ju.named_models_from_lens_system(system)
#     report = ju.profile_tree(named, root=sky(system), grad=True)
#     print(report)
