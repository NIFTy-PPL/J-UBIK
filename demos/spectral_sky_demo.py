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
# %% [markdown]
# ```{math}
# :label: eq-diffuse-mf
# I^{\mathrm{diff}}(\mathbf{x}, \nu)
# = I^{\mathrm{diff}}(\mathbf{x}, \nu_{\mathrm{ref}})
#   \left(\frac{\nu}{\nu_{\mathrm{ref}}}\right)^{\alpha(\mathbf{x})}
#   I_{\delta}(\mathbf{x}, \nu)\, .
# ```
#
# *Eq. (1): Spatio-spectral diffuse model used in this demo.*  
# We will refer to this as {eq}`eq-diffuse-mf` below.
# The idea behind this model is that the reference frequency sky brightness distribution is set by
#
# $$I^{\mathrm{diff}}(\mathbf{x}, \nu_{\mathrm{ref}}).$$
#
# Spectral deviations are then modeled by a power law with spectral index $\alpha(\mathbf{x})$.  
# Deviations from the power law are assumed to be spatially correlated and are modeled by the term
# $I_{\delta}(\mathbf{x}, \nu)$ in Eq. (1).  
# The resulting model is a product between a spatially correlated reference frequency sky brightness distribution
# and a spatially correlated spectral behavior which needs to be specified by the user.
# In the following, we show how to specify the spatial and spectral priors for this model.

# %% [markdown]
# ## Model Settings
# Here, we specify the settings for the model.

# %% [markdown]
# ### Spatial and spectral grids
# By setting the `shape`, `distances`, and `freqs` parameters, we can specify the spatial and spectral grids for the model.
# The reference frequency is given by `freqs[reference_frequency_index]`.
# While the spatial grid has to be a regular grid, the spectral input can take any spacing.  
# However we note that the model assumes that the frequencies are given as logarithmic frequencies, i.e. `freqs` should be $\log\nu$ from Eq. (1).

# %%
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
# The zero mode prior specifies a prior on the overall brightness of the model at the reference frequency.
# The prior is specified by a mean and standard deviation in log brightness units.

# %%
zero_mode_settings = (-3.0, 0.1)

# %% [markdown]
# #### Spatial Amplitude: Option 1 — Matérn kernel
# The spatial amplitude prior specifies a prior on the correlation structure of the brightness distribution
# $I^{\mathrm{diff}}(\mathbf{x}, \nu_{\mathrm{ref}})$ at the reference frequency.
#
# With a **Matérn kernel**, the correlation structure is parametrized by:
# - `scale`: overall amplitude of the fluctuations.  
# - `cutoff`: effective correlation length (in spatial units).  
# - `loglogslope`: slope of the log–log power spectrum at high $k$ (controls small-scale power).  
#
# For more details on how to specify the Matérn kernel, see
# [this notebook](https://ift.pages.mpcdf.de/nifty/user/a_correlated_field.html).
#
# *Note:* The Matérn kernel spatial amplitude model is here commented out below in favor of Option 2.

# %%
# amplitude_settings = dict(
#     scale=(0.4, 0.02),
#     cutoff=(0.1, 0.01),
#     loglogslope=(-4, 0.1),
# )
# amplitude_model = "matern"

# %% [markdown]
# #### Spatial Amplitude: Option 2 — Non-parametric correlated field model
# The non-parametric spatial amplitude model provides additional flexibility compared to the Matérn kernel.  
# Instead of committing to a fixed functional form, the correlation structure is encoded by a log–log power spectrum
# with an average slope and allowed fluctuations around it.  
# This is useful when the field contains both smooth large-scale structures and localized fine-scale features.
#
# **Hyperparameters:**
# - `fluctuations`: typical amplitude of variations around the average slope.  
# - `loglogavgslope`: average spectral slope in log–log space (more negative → less small-scale power).  
# - `flexibility`, `asperity`: optional shape controls (left `None` here for a smooth default).
#
# In practice, try a few nearby values of `loglogavgslope` (e.g., −3.5…−4.5) to match the spatial roughness you expect.
# For more details on how to set the correlated field hyperparameters, we refer again to [this notebook](https://ift.pages.mpcdf.de/nifty/user/a_correlated_field.html).

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
# Analogous to the spatial case, the **spectral amplitude** prior encodes correlation across **log-frequency**.  
# Choosing a non-parametric correlated field again lets the data inform how smoothly intensities co-vary with $\log \nu$
# without enforcing a single correlation length in frequency.
#
# **Hyperparameters:**
# - `fluctuations`: variability around the mean slope across frequency.  
# - `loglogavgslope`: average slope of the log–log spectrum along the frequency axis.  
# - `flexibility`, `asperity`: optional shape controls (left `None` for simplicity).
#
# A slope near −2 encourages smooth spectral behavior while still allowing moderate curvature.
#
# We note that the spectral amplitude can also be set to `None`.
# In this case, the spectral amplitude is assumed to be the same as the spatial amplitude, 
# i.e. the spatial correlation structure is the same along different frequency as at the reference frequency.
# We stress that this is rarely the case in practice, but it might be a good approximation in special cases or in the low signal-to-noise regime.

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
# The spectral index $\alpha(\mathbf{x})$ sets the power-law scaling in Eq.~(\ref{eq-diffuse-mf}):  
#
# $$I^{\mathrm{diff}}(\mathbf{x}, \nu) \propto
# \left(\frac{\nu}{\nu_{\mathrm{ref}}}\right)^{\alpha(\mathbf{x})}.$$
#
# We model $\alpha(\mathbf{x})$ as a correlated Gaussian field with:
# - `mean`: global average spectral index (e.g., −1.0).  
# - `fluctuations`: strength of spatial variability around the mean.
#
# Negative means correspond to spectra that dim with increasing frequency, which is common for many diffuse components.

# %%
spectral_idx_settings = dict(
    mean=(-1.0, 0.05),
    fluctuations=(0.1, 1.0e-2),
)

# %% [markdown]
# #### Deviations from Power Law
# Real spectra rarely follow a perfect power law.  
# The factor $I_{\delta}(\mathbf{x}, \nu)$ models **frequency-dependent departures** that remain spatially correlated.
#
# In this demo:
# - `process="wiener"`: deviations follow a (log-frequency) Wiener process, providing smooth, cumulative wiggles.  
# - `sigma`: typical amplitude of those deviations.
#
# This keeps spectra smooth overall while allowing coherent, frequency-dependent structure.

# %%
deviations_settings = dict(
    process="wiener",
    sigma=(0.2, 0.08),
)