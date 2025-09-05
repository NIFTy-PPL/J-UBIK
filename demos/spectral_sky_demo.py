#!/usr/bin/env python3

# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import jax.numpy as jnp
from jax import random
import jubik0 as ju

# %% [markdown]
# # Multi-Frequency Spectral Sky Model Demo
# This script sets up and visualizes a spectral sky model using `jubik0`.<br>
# Specifically, this model implements the spatio-spectral diffuse model described in
# [Guardiani et&nbsp;al., 2025](https://arxiv.org/abs/2506.20758).
#
# <a id="eq-diffuse-mf"></a>
# $$\begin{aligned}
# I^{\mathrm{diff}}(\mathbf{x}, \nu)
# \;=\;
# I^{\mathrm{diff}}(\mathbf{x}, \nu_{\mathrm{ref}})
# \left(\frac{\nu}{\nu_{\mathrm{ref}}}\right)^{\alpha(\mathbf{x})}
# \, I_{\delta}(\mathbf{x}, \nu)\, .
# \end{aligned}$$
#
# *Eq. (1): Spatio-spectral diffuse model used in this demo.*  
# We will refer to this as [Eq. (1)](#eq-diffuse-mf) below.  
# The idea behind this model is that the reference frequncy sky brightness distribution is set by
# $$I^{\mathrm{diff}}(\mathbf{x}, \nu_{\mathrm{ref}}).$$
# Spectral deviations are then modeled by a power law with spectral index $\alpha(\mathbf{x})$.  
# Deviations from the power law are assumed to be spatially correlated and are modeled by the term $I_{\delta}(\mathbf{x}, \nu)$ in Eq. (1).
# The resulting model is a product between a spatially correlated reference frequency sky brightness distribution
# and a spatially correlated spectral behavior which needs to be specified by the user.
# In the following, we show how to specify the spatial and spectral priors for this model.

# %% [markdown]
# ## Model Settings
# Here, we specify the settings for the model.

# %% [markdown]
# ### Spatial and spectral grids
# By setting the `shape`, `distances`, and `freqs` parameters, we can specify the spatial and spectral grids for the model.
# While the spatial grid has to be a regular grid, the spectral input can take any spacing.  
# However we note that the model assumes that the frequencies are given as logarithmic frequencies, i.e. `freqs` should be $\log\nu$ from Eq. (1).

shape = (256,) * 2
distances = 0.1
freqs, reference_frequency_index = jnp.array((0.1, 1.5, 2, 10)), 1

# %% [markdown]
# ### Random Seed
# We also need to set a random seed for the model. 
# This defines the specific realization of the model that we will draw in the following examples.

# %%
seed = 42
key = random.PRNGKey(seed)

# %% [markdown]
# ### Prior Settings
# In the following, we specify the prior settings for the model.

# %% [markdown]
# #### Zero mode
zero_mode_settings = (-3.0, 0.1)

# %% [markdown]
# #### Spatial Amplitude: Option 1 — Matérn kernel
# The spatial amplitude prior specifies the a prior on the correlation structure of the brightness distribution $I^{\mathrm{diff}}(\mathbf{x}, \nu_{\mathrm{ref}})$ at the reference frequency.
# It is possible to parametrize the spatial amplitude using a Matérn kernel.
# This is done by setting the `scale`, `cutoff`, and `loglogslope` priors in the `amplitude_settings` dictionary.
# For more details on how to specify the Matérn kernel, see [this notebook](https://ift.pages.mpcdf.de/nifty/user/a_correlated_field.html).
# (The Matérn kernel spatial amplitude model is here commented out below in favor of Option 2)

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
