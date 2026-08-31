from importlib import import_module

import jax.numpy as jnp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.data import AntennaPositions, Observation
from jubik.polarization import Polarization, PolarizationType
from jubik.wcs import WcsAstropy


dirty_image_module = import_module("jubik.instruments.resolve.dirty_image")


def test_dirty_image_preserves_off_center_source_position():
    shape = (24, 24)
    frequency = np.array([1.4e9])
    grid = ju.Grid(
        WcsAstropy(
            center=SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg),
            shape=shape,
            fov=u.Quantity((1.0, 1.0), u.deg),
        ),
        ju.Color.from_central_frequencies(frequency),
        polarization=PolarizationType.I,
    )

    rng = np.random.default_rng(42)
    n_rows = 256
    radius = 80.0 * np.sqrt(rng.uniform(size=n_rows))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_rows)
    uvw = np.column_stack(
        (radius * np.cos(angle), radius * np.sin(angle), np.zeros(n_rows))
    )
    antenna_positions = AntennaPositions(uvw)
    visibility_shape = (1, n_rows, frequency.size)
    empty_observation = Observation(
        antenna_positions,
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

    source_position = (7, 15)
    sky = np.zeros(grid.shape)
    sky[0, 0, 0, *source_position] = 1.0 / grid.spatial.dvol.to(u.rad**2).value
    visibilities = np.asarray(sky_to_vis(jnp.asarray(sky)))
    observation = Observation(
        antenna_positions,
        vis=visibilities,
        weight=np.ones(visibility_shape, dtype=np.float64),
        polarization=PolarizationType.I.get_legacy_polarization(),
        freq=frequency,
    )

    dirty = dirty_image_module.dirty_image(
        observation,
        grid,
        backend_settings=backend_settings,
    ).value[0, 0, 0]

    assert np.unravel_index(np.argmax(dirty), dirty.shape) == source_position


def test_uniform_weights_fills_every_polarization(monkeypatch):
    observation = Observation(
        AntennaPositions(np.zeros((1, 3))),
        vis=np.ones((2, 1, 1), dtype=np.complex128),
        weight=np.array([[[2.0]], [[3.0]]]),
        polarization=Polarization([8, 5]),
        freq=np.array([1.0]),
    )

    def fake_uvw_density(eff_u, eff_v, sky_grid, weights):
        histogram = np.ones((1, 1)) if weights is None else np.array([[weights[0, 0]]])
        edges = np.array([-1.0, 1.0])
        return histogram, edges, edges

    monkeypatch.setattr(dirty_image_module, "uvw_density", fake_uvw_density)

    result = dirty_image_module.uniform_weights(observation, sky_grid=object())

    assert_allclose(result, observation.weight_val)
