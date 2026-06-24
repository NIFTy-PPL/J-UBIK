# %% [markdown]
# # Multi-Frequency Spectral Sky Model Demo on HEALPix
#
# This demo draws one prior sample from the multi-frequency spectral sky model
# on a HEALPix sphere and visualizes the resulting maps with `healpy.mollview`.
#
# The model is
#
#     I(x, nu) = I(x, nu_ref) * exp[alpha(x) * (log nu - log nu_ref)] * I_delta(x, nu)
#
# In this HEALPix version:
# - `shape=(nside,)`
# - `harmonic_type="spherical"`
# - model outputs have shape `(n_frequencies, 12*nside**2)`

# %%
import jax.numpy as jnp
from jax import random
import jubik as ju


# %% [markdown]
# ## Grid and frequencies

# %%
nside = 32
shape = (nside,)
distances = (1.0,)
freqs = jnp.array((0.1, 1.5, 2.0, 10.0))
reference_frequency_index = 1


# %% [markdown]
# ## Random seed

# %%
seed = 42
key = random.PRNGKey(seed)


# %% [markdown]
# ## Prior settings

# %%
zero_mode_settings = (-3.0, 0.1)

spatial_amplitude_settings = dict(
    fluctuations=(1.0, 0.02),
    loglogavgslope=(-4.0, 0.1),
    flexibility=None,
    asperity=None,
)

spectral_amplitude_settings = dict(
    fluctuations=(1.0, 0.02),
    loglogavgslope=(-2.0, 0.1),
    flexibility=None,
    asperity=None,
)

spectral_index_settings = dict(
    mean=(-1.0, 0.05),
    fluctuations=(0.1, 1.0e-2),
)

deviations_settings = dict(
    process="wiener",
    sigma=(0.2, 0.08),
)


# %% [markdown]
# ## Build HEALPix spectral sky model

# %%
mf_model = ju.build_simple_spectral_sky(
    prefix="healpix_demo",
    shape=shape,
    distances=distances,
    log_frequencies=freqs,
    reference_frequency_index=reference_frequency_index,
    zero_mode_settings=zero_mode_settings,
    spatial_amplitude_settings=spatial_amplitude_settings,
    spectral_index_settings=spectral_index_settings,
    spectral_amplitude_settings=spectral_amplitude_settings,
    deviations_settings=deviations_settings,
    spatial_amplitude_model="non_parametric",
    spectral_amplitude_model="non_parametric",
    harmonic_type="spherical",
    sht_nthreads=1,
)

random_pos = mf_model.init(key)


# %% [markdown]
# ## Plot reference-frequency distribution

# %%
ju.plot_healpix_result(
    mf_model.reference_frequency_distribution(random_pos),
    n_rows=1,
    n_cols=1,
    figsize=(10, 6),
    title="Reference frequency distribution",
    unit="intensity",
)


# %% [markdown]
# ## Plot spectral index

# %%
ju.plot_healpix_result(
    mf_model.spectral_index_distribution(random_pos),
    n_rows=1,
    n_cols=1,
    figsize=(10, 6),
    title="Spectral index distribution",
    unit="alpha",
)


# %% [markdown]
# ## Plot spectral deviations

# %%
ju.plot_healpix_result(
    mf_model.spectral_deviations_distribution(random_pos),
    n_rows=2,
    n_cols=2,
    figsize=(12, 7),
    title=[f"Deviation at log nu={float(nu):.2f}" for nu in freqs],
    common_colorbar=True,
)


# %% [markdown]
# ## Plot full multi-frequency sky realization

# %%
ju.plot_healpix_result(
    mf_model(random_pos),
    n_rows=2,
    n_cols=2,
    figsize=(12, 7),
    title=[f"Sky at log nu={float(nu):.2f}" for nu in freqs],
    unit="intensity",
    common_colorbar=True,
)
