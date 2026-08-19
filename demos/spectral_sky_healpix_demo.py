# %% [markdown]
# # Multi-Frequency Spectral Sky Model Demo on HEALPix
#
# This demo draws prior samples from two HEALPix-compatible multi-frequency
# sky components:
#
# 1. a correlated diffuse spectral-product sky, and
# 2. an uncorrelated inverse-gamma point-source sky.
#
# The diffuse component is spatially correlated and uses the HEALPix spherical
# harmonic transform. The point-source component is uncorrelated in space, so
# for HEALPix it only needs the correct pixel-domain shape:
#
# ```python
# shape = (nside,)  ->  npix = 12 * nside**2
# ```
#
# The diffuse model is
#
# $$
# I^{\mathrm{diff}}(x,\nu)
# =
# I^{\mathrm{diff}}(x,\nu_{\mathrm{ref}})
# \exp\left[
#     \alpha^{\mathrm{diff}}(x)
#     \left(\log\nu-\log\nu_{\mathrm{ref}}\right)
# \right]
# I^{\mathrm{diff}}_{\delta}(x,\nu).
# $$
#
# The point-source model is
#
# $$
# I^{\mathrm{ps}}(x,\nu)
# =
# I^{\mathrm{ps}}(x,\nu_{\mathrm{ref}})
# \exp\left[
#     \gamma^{\mathrm{ps}}(x)
#     \left(\log\nu-\log\nu_{\mathrm{ref}}\right)
#     +
#     \delta^{\mathrm{ps}}(x,\nu)
# \right].
# $$
#
# The reference-frequency point-source map is drawn independently per pixel from
# an inverse-gamma prior:
#
# $$
# I^{\mathrm{ps}}(x,\nu_{\mathrm{ref}})
# \sim
# \mathrm{IG}(\alpha_{\mathrm{IG}}, q).
# $$
#
# The total sky is simply
#
# $$
# I^{\mathrm{tot}}(x,\nu)
# =
# I^{\mathrm{diff}}(x,\nu)
# +
# I^{\mathrm{ps}}(x,\nu).
# $$
#
# We'll build this up in three acts: first the diffuse component on its own,
# then the point-source component on its own, then combine them into the
# total sky.

# %%
import jax.numpy as jnp
from jax import random
import jubik as ju


# %% [markdown]
# ## Setup
#
# ### Grid and frequencies

# %%
nside = 128
shape = (nside,)
distances = (1.0,)

log_frequencies = jnp.array((0.1, 1.0, 1.5, 2.0, 5.0, 7.0, 10.0, 11.0))
reference_frequency_index = 2

# %% [markdown]
# ### Random seed

# %%
seed = 42
key = random.PRNGKey(seed)
key_diffuse, key_points = random.split(key, 2)


# %% [markdown]
# ## Diffuse component
#
# ### Prior settings

# %%
zero_mode_settings = (-3.0, 0.1)

spatial_amplitude_settings = dict(
    fluctuations=(1.0, 0.02),
    loglogavgslope=(-2.0, 0.1),
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
    fluctuations=(1.0, 1.0e-1),
)

deviations_settings = dict(
    process="wiener",
    sigma=(0.2, 0.08),
)

# %% [markdown]
# ### Build the model

# %%
diffuse_model = ju.build_simple_spectral_sky(
    prefix="healpix_diffuse_demo",
    shape=shape,
    distances=distances,
    log_frequencies=log_frequencies,
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

diffuse_pos = diffuse_model.init(key_diffuse)

# %% [markdown]
# ### Results

# %%
diffuse_sky = diffuse_model(diffuse_pos)
print("diffuse_sky shape:", diffuse_sky.shape)

# %%
ju.plot_healpix_result(
    diffuse_model.reference_frequency_distribution(diffuse_pos),
    n_rows=1,
    n_cols=1,
    figsize=(10, 5.2),
    title="Diffuse reference-frequency distribution",
    unit="intensity",
    logscale=True,
    common_colorbar=False,
)

# %%
ju.plot_healpix_result(
    diffuse_model.spectral_index_distribution(diffuse_pos),
    n_rows=1,
    n_cols=1,
    figsize=(10, 5.5),
    title="Diffuse spectral index distribution",
    unit="alpha",
    logscale=False,
    common_colorbar=False,
)

# %%
diffuse_spectral_deviations = diffuse_model.spectral_deviations_distribution(
    diffuse_pos
)

ju.plot_healpix_result(
    diffuse_spectral_deviations,
    n_rows=2,
    n_cols=4,
    figsize=(13, 6.5),
    title=[
        f"Diffuse deviation\nlog nu={float(nu):.2f}"
        for nu in log_frequencies
    ],
    unit="log deviation",
    logscale=False,
    common_colorbar=False,
)

# %%
ju.plot_healpix_result(
    diffuse_sky,
    n_rows=2,
    n_cols=4,
    figsize=(13, 6.5),
    title=[
        f"Diffuse sky\nlog nu={float(nu):.2f}"
        for nu in log_frequencies
    ],
    unit="intensity",
    logscale=True,
    common_colorbar=False,
)


# %% [markdown]
# ## Point-source component
#
# ### Prior settings
#
# The point-source model is uncorrelated in space. For
# `harmonic_type="spherical"`, the builder interprets `shape=(nside,)` as a
# HEALPix grid and expands it internally to `(12*nside**2,)`.
#
# The inverse-gamma parameters control the reference-frequency brightness field.
# The spectral-index mean is per pixel by default when `shared=False`.

# %%
ps_alpha = 2
ps_q = 0.08

ps_spectral_settings = dict(
    mean=(-1.0, 0.3),
    deviations=dict(process="wiener", sigma=(0.2, 0.08)),
    shared=False,
)

# %% [markdown]
# ### Build the model

# %%
point_source_model = ju.build_mf_invgamma_sky(
    prefix="healpix_points_demo",
    alpha=ps_alpha,
    q=ps_q,
    shape=shape,
    log_frequencies=log_frequencies,
    reference_frequency_index=reference_frequency_index,
    spectral_settings=ps_spectral_settings,
    harmonic_type="spherical",
)

point_source_pos = point_source_model.init(key_points)

# %% [markdown]
# ### Results

# %%
point_source_sky = point_source_model(point_source_pos)
print("point_source_sky shape:", point_source_sky.shape)

# %%
point_reference_map = point_source_model.reference_frequency_distribution(
    point_source_pos
)

ju.plot_healpix_result(
    point_reference_map,
    n_rows=1,
    n_cols=1,
    figsize=(10, 5.5),
    title="Point-source reference-frequency distribution",
    unit="intensity",
    logscale=True,
    vmin=1.0e-1,
    #percentile=(1.0, 99.9),
    common_colorbar=False,
)

# %% [markdown]
# If `shared=True`, this is a scalar. If `shared=False`, this is a HEALPix map.

# %%
point_spectral_index = point_source_model.spectral_index_distribution(
    point_source_pos
)

if point_spectral_index.shape == ():
    point_spectral_index = jnp.full((12 * nside**2,), point_spectral_index)

ju.plot_healpix_result(
    point_spectral_index,
    n_rows=1,
    n_cols=1,
    figsize=(10, 5.5),
    title="Point-source spectral index distribution",
    unit="gamma",
    logscale=False,
    common_colorbar=False,
)

# %%
point_spectral_deviations = point_source_model.spectral_deviations_distribution(
    point_source_pos
)

if point_spectral_deviations is not None:
    ju.plot_healpix_result(
        point_spectral_deviations,
        n_rows=2,
        n_cols=4,
        figsize=(13, 6.5),
        title=[
            f"Point-source deviation\nlog nu={float(nu):.2f}"
            for nu in log_frequencies
        ],
        unit="log deviation",
        logscale=False,
        common_colorbar=False,
    )

# %%
ju.plot_healpix_result(
    point_source_sky,
    n_rows=2,
    n_cols=4,
    figsize=(13, 6.5),
    title=[
        f"Point-source sky\nlog nu={float(nu):.2f}"
        for nu in log_frequencies
    ],
    unit="intensity",
    logscale=True,
    vmin=1.0e-1,
    #percentile=(1.0, 99.9),
    common_colorbar=False,
)


# %% [markdown]
# ## Total sky: diffuse + point sources

# %%
total_sky = diffuse_sky + point_source_sky
print("total_sky shape:", total_sky.shape)
print("expected shape:", (log_frequencies.shape[0], 12 * nside**2))

# %%
ju.plot_healpix_result(
    total_sky,
    n_rows=2,
    n_cols=4,
    figsize=(13, 6.5),
    title=[
        f"Total sky\nlog nu={float(nu):.2f}"
        for nu in log_frequencies
    ],
    unit="intensity",
    logscale=True,
    #percentile=(1.0, 99.9),
    common_colorbar=False,
)

# %%
