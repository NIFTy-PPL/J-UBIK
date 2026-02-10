# %% [markdown]
# # Multi-Frequency Point Source Sky Model Demo
# This script sets up and visualizes the point-source sky model in `jubik`.
# It is the uncorrelated (pixel-wise) counterpart to the diffuse correlated model
# in `spectral_sky_demo.py`.
#
# ```{math}
# :label: eq-ps-mf
# I^{\mathrm{ps}}(\mathbf{x}, \nu)
# = I^{\mathrm{ps}}(\mathbf{x}, \nu_{\mathrm{ref}})
#   \left(\frac{\nu}{\nu_{\mathrm{ref}}}\right)^{\alpha(\mathbf{x})}
#   I_{\delta}(\mathbf{x}, \nu)\, .
# ```
#
# *Eq. (1): Point-source spatio-spectral model used in this demo.*
#
# **Key idea:** the reference-frequency map is *independent across pixels* and
# drawn from an inverse-gamma (IG) distribution.
# The prior factorizes as
#
# ```{math}
# \mathrm{IG}\big(I^{\mathrm{ps}}(\cdot, \nu_{\mathrm{ref}})\,|\,\alpha, q\big)
# = \prod_{\mathbf{x}} \mathrm{IG}\big(I^{\mathrm{ps}}(\mathbf{x}, \nu_{\mathrm{ref}})\,|\,\alpha, q\big).
# ```
#
# So there is **one IG distribution per pixel** (not per dimension), and all
# pixels share the same parameters when `alpha` and `q` are scalars.
#
# We denote the inverse-gamma density by $\mathrm{IG}(x\,|\,\alpha,q)$.
# In the NIFTy.re parameterization used by `jubik`, this density is
#
# ```{math}
# \mathrm{IG}(x\,|\,\alpha,q) = \frac{q^{\alpha}}{\Gamma(\alpha)}\,x^{-(\alpha+1)}\,\exp(-q/x),\quad x>0.
# ```
#
# This is heavy-tailed and strictly positive, which makes it a natural prior for
# sparse point-source fields with many faint pixels and a few bright outliers.
# If $\alpha>1$, the mean is $\mathbb{E}[x]=q/(\alpha-1)$; if $\alpha>2$, the
# variance is $q^2/((\alpha-1)^2(\alpha-2))$.

# %% [markdown]
# ## Model Settings
# We specify the spatial and spectral grids. As in the diffuse demo, the model
# expects **log-frequencies**.

# %%
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import jubik as ju

shape = (128,) * 2
log_frequencies = jnp.array((0.1, 1.5, 2.0, 10.0))
reference_frequency_index = 1

# %% [markdown]
# ### Random Seed
# As before, we fix a PRNG seed for reproducible samples.

# %%
seed = 42
key = random.PRNGKey(seed)

# %% [markdown]
# ## Prior Settings
#
# ### Inverse-Gamma Reference Map
# The IG prior sets the point-source brightness distribution at the reference
# frequency. Larger `alpha` suppresses extremely bright pixels; `q` sets the
# overall scale. Choose `alpha > 1` if you want a finite mean.

# %%
alpha = 2.5
q = 0.08

# %% [markdown]
# ### Spectral Index and Deviations
# The spectral index is the exponent of the power-law scaling:
# ```{math}
# I(\nu) \propto (\nu/\nu_{\mathrm{ref}})^{\alpha}.
# ```
# Negative values mean the spectrum dims with increasing frequency.
#
# Because point sources are spatially uncorrelated, we default to a **per-pixel**
# spectral index map (`shared=False`). This means each pixel draws its own
# spectral index from the prior. Set `shared=True` to enforce a single global
# slope for all pixels.
#
# Deviations capture smooth, frequency-dependent departures from a pure power
# law. Here they are modeled as a Wiener process in log-frequency for each
# pixel. Since the point-source model is uncorrelated in space, these deviations
# are also independent across pixels.

# %%
spectral_settings = dict(
    mean=(-1.0, 0.3),
    deviations=dict(process="wiener", sigma=(0.2, 0.08)),
    shared=False,
)

# %% [markdown]
# ## Build the Model

# %%
ps_model = ju.build_mf_invgamma_sky(
    prefix="ps",
    alpha=alpha,
    q=q,
    shape=shape,
    log_frequencies=log_frequencies,
    reference_frequency_index=reference_frequency_index,
    spectral_settings=spectral_settings,
)

# %% [markdown]
# ## Draw a Prior Sample

# %%
random_pos = ps_model.init(key)
reference_map = ps_model.reference_frequency_distribution(random_pos)

# %% [markdown]
# ## Plot Results
#
# ### Reference-Frequency Map
# The point-source field is spatially *uncorrelated* and strictly positive.
# The heavy tail of the IG prior produces a few bright pixels amid many faint
# ones.

# %%
ju.plot_result(
    reference_map,
    n_rows=1,
    n_cols=1,
    figsize=(7, 5),
    title="Point-source reference map",
    logscale=True,
)

# %% [markdown]
# Additionally, we can clip the color scale to mimic a minimum brightness
# detection threshold (visualization only). Here we clip at `1e-1` for display.

# %%
threshold = 1e-1
ju.plot_result(
    reference_map,
    n_rows=1,
    n_cols=1,
    figsize=(7, 5),
    title="Point-source reference map (clipped for display)",
    logscale=True,
    vmin=threshold,
)

# %% [markdown]
# ### Pixel Histogram (all pixels)
# Since pixels are IID, the histogram of all pixels approximates the IG prior.

# %%
plt.figure(figsize=(7, 4))
plt.hist(reference_map.ravel(), bins=200, log=True)
plt.xlabel("Reference-frequency brightness")
plt.ylabel("Pixel count (log scale)")
plt.title("Inverse-gamma prior over pixels")
plt.tight_layout()

# %% [markdown]
# ### Spectral Index Map
# If `shared=False`, this is a spatial map of spectral indices; if `shared=True`,
# it is a single scalar value broadcast across the field.

# %%
spectral_index_map = ps_model.spectral_index_distribution(random_pos)
if spectral_index_map.shape == ():
    spectral_index_map = jnp.full(shape, spectral_index_map)

ju.plot_result(
    spectral_index_map,
    n_rows=1,
    n_cols=1,
    figsize=(7, 5),
    title="Spectral index mean (alpha)",
)

# %% [markdown]
# ### Spectral Deviations (log-space)
# These are additive deviations in log-space, so they can be positive or
# negative. We therefore keep `logscale=False` for visualization.

# %%
spectral_deviations = ps_model.spectral_deviations_distribution(random_pos)
if spectral_deviations is not None:
    ju.plot_result(
        spectral_deviations,
        n_rows=1,
        n_cols=log_frequencies.shape[0],
        figsize=(15, 4),
        title="Spectral deviations",
        logscale=False,
    )

# %% [markdown]
# ### Multifrequency Realization
# A final multifrequency sky realization can be visualized by plotting the
# point-source model at a random latent position

# %%
ju.plot_result(
    ps_model(random_pos),
    n_rows=1,
    n_cols=log_frequencies.shape[0],
    figsize=(15, 4),
    title="Point-source multi-frequency realization",
    logscale=True,
)

# %% [markdown]
# ## Combining Diffuse + Point Sources (sketch)
# The total sky model is typically a sum of the diffuse and point-source
# components. You can combine them with `ju.add_models`:
#
# ```python
# diffuse = ju.build_simple_spectral_sky(...)
# point_sources = build_mf_invgamma_sky(...)
# sky = ju.add_models(diffuse, point_sources)
# ```
#
# See `spectral_sky_demo.py` for the full diffuse-model settings.
