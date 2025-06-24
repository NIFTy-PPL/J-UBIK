#!/usr/bin/env python3

# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Multi-Frequency Spectral Sky Model Demo
# This script sets up and visualizes a spectral sky model using `jubik0`.

# %%
import jax.numpy as jnp
from jax import random
import jubik0 as ju

# %% [markdown]
# ### Model Settings

# %%
# Shape and frequency setup
shape = (256,) * 2
distances = 0.1
freqs, reference_frequency_index = jnp.array((0.1, 1.5, 2, 10)), 1

# %% [markdown]
# ### Random Seed

# %%
seed = 42
key = random.PRNGKey(seed)

# %% [markdown]
# ### Prior Settings

# %%
# Zero mode
zero_mode_settings = (-3.0, 0.1)

# %% [markdown]
# #### Spatial Amplitude: Option 1 — Matérn kernel
# (Commented out below in favor of Option 2)

# %%
# amplitude_settings = dict(
#     scale=(0.4, 0.02),
#     cutoff=(0.1, 0.01),
#     loglogslope=(-4, 0.1),
# )
# amplitude_model = "matern"

# %% [markdown]
# #### Spatial Amplitude: Option 2 — Non-parametric correlated field model

# %%
amplitude_settings = dict(
    fluctuations=(1.0, 0.02),
    loglogavgslope=(-4, 0.1),
    flexibility=None,
    asperity=None,
)
amplitude_model = "non_parametric"

# %% [markdown]
# #### Spectral Amplitude

# %%
spectral_amplitude_settings = dict(
    fluctuations=(1.0, 0.02),
    loglogavgslope=(-2.0, 0.1),
    flexibility=None,
    asperity=None,
)
spectral_amplitude_model = "non_parametric"

# %% [markdown]
# #### Spectral Index

# %%
spectral_idx_settings = dict(
    mean=(-1.0, 0.05),
    fluctuations=(0.1, 1.0e-2),
)

# %% [markdown]
# #### Deviations from Power Law

# %%
deviations_settings = dict(
    process="wiener",
    sigma=(0.2, 0.08),
)

# %% [markdown]
# ### Build Model

# %%
mf_model = ju.build_simple_spectral_sky(
    prefix="test",
    shape=shape,
    distances=distances,
    log_frequencies=freqs,
    reference_frequency_index=reference_frequency_index,
    zero_mode_settings=zero_mode_settings,
    spatial_amplitude_settings=amplitude_settings,
    spectral_index_settings=spectral_idx_settings,
    deviations_settings=deviations_settings,
    spatial_amplitude_model=amplitude_model,
    spectral_amplitude_settings=spectral_amplitude_settings,
    spectral_amplitude_model=spectral_amplitude_model,
)

# %% [markdown]
# ### Draw Prior Sample Realization

# %%
random_pos = mf_model.init(key)

# %% [markdown]
# ### Plot Results

# %%
ju.plot_result(
    mf_model.reference_frequency_distribution(random_pos),
    n_rows=1,
    n_cols=1,
    figsize=(15, 5),
    title="Reference frequency distribution",
)

# %%
ju.plot_result(
    mf_model.spectral_index_distribution(random_pos),
    n_rows=1,
    n_cols=1,
    figsize=(15, 5),
    title="Spectral index distribution",
)

# %%
ju.plot_result(
    mf_model.spectral_deviations_distribution(random_pos),
    n_rows=1,
    n_cols=freqs.shape[0],
    figsize=(15, 5),
    title="Spectral deviations distribution",
)

# %%
ju.plot_result(
    mf_model(random_pos),
    n_rows=1,
    n_cols=freqs.shape[0],
    figsize=(15, 5),
    title="MF model realization",
)