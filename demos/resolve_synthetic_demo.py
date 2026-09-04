# %% [markdown]
# # Synthetic radio-interferometry demo
#
# This demo builds a small radio-interferometric imaging problem without
# downloading a measurement set. We generate baseline coordinates and a known
# Stokes-I sky, apply J-UBIK's Resolve response, add Gaussian visibility noise,
# and reconstruct the sky from the resulting synthetic observation.
#
# The example deliberately uses one frequency channel so that it executes
# quickly as part of the documentation build. The same response works with the
# multi-frequency prior introduced in the
# {doc}`spectral sky demo <spectral_sky_demo>`.

# %% [markdown]
# ## Imports and reproducibility
#
# Resolve's DUCC response uses 64-bit coordinates, so we enable JAX's 64-bit
# mode. Splitting one random key keeps the simulated data and reconstruction
# reproducible.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from jax import random

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.polarization import PolarizationType
from jubik.wcs import WcsAstropy


jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)
key, position_key, inference_key = random.split(key, 3)
rng = np.random.default_rng(seed)

# %% [markdown]
# ## Sky grid and synthetic baseline coverage
#
# An interferometer samples Fourier components of the sky at coordinates set
# by its antenna baselines. For this compact example we draw 256 baseline
# vectors inside a disk with a maximum length of 80 metres. The observation
# only needs the baseline coordinates for imaging; antenna labels and times
# become necessary when calibration effects are inferred as well.

# %%
shape = (24, 24)
frequency = np.array([1.4e9])
fov = u.Quantity((1.0, 1.0), u.deg)

spatial_grid = WcsAstropy(
    center=SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg),
    shape=shape,
    fov=fov,
)
spectral_grid = ju.Color.from_central_frequencies(frequency)
grid = ju.Grid(spatial_grid, spectral_grid, polarization=PolarizationType.I)

n_rows = 256
radius = 80.0 * np.sqrt(rng.uniform(size=n_rows))
angle = rng.uniform(0.0, 2.0 * np.pi, size=n_rows)
uvw = np.column_stack(
    (radius * np.cos(angle), radius * np.sin(angle), np.zeros(n_rows))
)
antenna_positions = rve.AntennaPositions(uvw)

visibility_shape = (1, n_rows, frequency.size)
empty_observation = rve.Observation(
    antenna_positions=antenna_positions,
    vis=np.zeros(visibility_shape, dtype=np.complex128),
    weight=np.ones(visibility_shape, dtype=np.float64),
    polarization=PolarizationType.I.get_legacy_polarization(),
    freq=frequency,
)

backend_settings = rve.parse.Ducc0Settings(
    epsilon=1.0e-7,
    do_wgridding=False,
    nthreads=1,
    verbosity=False,
)
sky_to_vis = rve.interferometry_response(
    empty_observation,
    grid,
    backend_settings=backend_settings,
)

# %% [markdown]
# The sampled Fourier coordinates are conventionally called the $uv$ coverage.
# Longer baselines constrain finer angular scales.

# %%
effective_uvw = empty_observation.effective_uvw()
plt.figure(figsize=(5, 5))
plt.scatter(effective_uvw[0, :, 0], effective_uvw[1, :, 0], s=8)
plt.xlabel(r"$u$ [wavelengths]")
plt.ylabel(r"$v$ [wavelengths]")
plt.title("Synthetic baseline coverage")
plt.gca().set_aspect("equal")
plt.show()

# %% [markdown]
# ## Generate synthetic visibilities
#
# The ground truth consists of two smooth emission components on top of a weak
# background. Applying the response produces complex visibilities at the
# sampled baselines. We then add independent complex Gaussian noise and store
# its inverse variance in the observation weights.

# %%
xx, yy = np.meshgrid(
    np.linspace(-1.0, 1.0, shape[0]),
    np.linspace(-1.0, 1.0, shape[1]),
    indexing="ij",
)
truth_image = (
    3.0e3
    + 5.0e4 * np.exp(-((xx + 0.28) ** 2 + (yy - 0.12) ** 2) / 0.07)
    + 3.5e4 * np.exp(-((xx - 0.30) ** 2 + (yy + 0.25) ** 2) / 0.025)
)
truth_sky = truth_image[None, None, None, :, :]
noiseless_visibilities = np.asarray(sky_to_vis(jnp.asarray(truth_sky)))

noise_std = 2.0e-2
noise = (
    noise_std
    / np.sqrt(2.0)
    * (rng.normal(size=visibility_shape) + 1j * rng.normal(size=visibility_shape))
)
visibilities = noiseless_visibilities + noise
weights = np.full(visibility_shape, noise_std**-2)

observation = rve.Observation(
    antenna_positions=antenna_positions,
    vis=visibilities,
    weight=weights,
    polarization=PolarizationType.I.get_legacy_polarization(),
    freq=frequency,
)

# %% [markdown]
# ## Inspect the dirty image
#
# The dirty image is the weighted adjoint response applied to the data. It is a
# useful quick-look image, but it contains sidelobes from the incomplete Fourier
# coverage and is not yet a Bayesian reconstruction.

# %%
dirty = rve.dirty_image(
    observation,
    grid,
    backend_settings=backend_settings,
).value[0, 0, 0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
images = (
    axes[0].imshow(truth_image.T, origin="lower"),
    axes[1].imshow(dirty.T, origin="lower"),
)
axes[0].set_title("Ground truth")
axes[1].set_title("Dirty image")
for ax, image in zip(axes, images):
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    fig.colorbar(image, ax=ax, shrink=0.8)
plt.show()

# %% [markdown]
# ## Build a correlated sky prior and likelihood
#
# We use the same positive correlated-field builder as the multi-frequency
# example. With one channel the spectral variation is inactive; adding more
# channels turns it into an inferred spatially varying spectrum.

# %%
diffuse_sky = ju.build_simple_spectral_sky(
    prefix="resolve_demo",
    shape=shape,
    distances=grid.spatial.distances.to(u.rad).value,
    log_frequencies=np.log(frequency),
    reference_frequency_index=0,
    zero_mode_settings=(np.log(2.0e4), 1.0),
    spatial_amplitude_settings={
        "fluctuations": (1.0, 0.2),
        "loglogavgslope": (-4.0, 0.5),
        "flexibility": None,
        "asperity": None,
    },
    spectral_index_settings={
        "mean": (-0.7, 0.2),
        "fluctuations": (0.2, 0.05),
    },
    spectral_amplitude_settings=None,
    deviations_settings=None,
)
sky = jft.Model(
    lambda x: diffuse_sky(x)[None, None, ...],
    domain=diffuse_sky.domain,
)


def signal_response(x):
    return sky_to_vis(sky(x))


likelihood = jft.Gaussian(
    observation.vis_val,
    observation.weight_val,
).amend(signal_response)

# %% [markdown]
# ## Reconstruct a maximum-a-posteriori sky
#
# For a fast documentation example we run a few maximum-a-posteriori (MAP)
# updates by setting `n_samples=0`. For uncertainty quantification, use one or
# more samples and the MGVI or geoVI settings demonstrated by the instrument
# pipeline examples.

# %%
position = jft.Vector(sky.init(position_key))
logging_was_disabled = jft.logger.disabled
jft.logger.disabled = True
try:
    samples, state = jft.optimize_kl(
        likelihood,
        position,
        key=inference_key,
        n_total_iterations=3,
        n_samples=0,
        kl_kwargs={
            "minimize_kwargs": {
                "name": "MAP",
                "maxiter": 5,
                "cg_kwargs": {"name": "MAP-CG", "maxiter": 30},
            }
        },
    )
finally:
    jft.logger.disabled = logging_was_disabled
reconstruction = np.asarray(sky(samples.pos))[0, 0, 0]

# %% [markdown]
# The reconstruction combines the visibility constraints with the correlated
# sky prior. This tiny problem is intended to demonstrate the complete data
# flow rather than provide a science-grade reconstruction.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
panels = (truth_image, dirty, reconstruction)
titles = ("Ground truth", "Dirty image", "MAP reconstruction")
for ax, panel, title in zip(axes, panels, titles):
    image = ax.imshow(panel.T, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    fig.colorbar(image, ax=ax, shrink=0.8)
plt.show()

# %% [markdown]
# ## Next steps
#
# A realistic workflow can replace the generated `Observation` with data loaded
# from disk and can add calibration parameters, flags, frequency averaging, and
# polarization. For multi-frequency imaging, construct the grid with several
# channel centres and use the spectral prior from the multi-frequency demo.
