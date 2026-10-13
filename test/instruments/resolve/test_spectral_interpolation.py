"""Spectral mapping on canonical radio cubes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from jubik.color import Color
from jubik.grid import Grid
from jubik.wcs.wcs_astropy import WcsAstropy
from jubik.instruments.resolve.mosaicing.spectral_interpolation import (
    SpectralInterpolator,
)


def grid(bounds):
    # Creates a Grid object with spatial grid of shape (3,2) and spectral grid from input bounds
    return Grid(
        WcsAstropy(
            center=SkyCoord(0 * u.deg, 0 * u.deg),
            shape=(3, 2),
            fov=(3 * u.arcsec, 2 * u.arcsec),
        ),
        Color(bounds * u.Hz),
    )


def test_irregular_centers_edges_jit_and_gradient():
    g = grid(np.array([0.0, 2.0, 6.0, 12.0]))  # centers 1, 4, 9
    op = SpectralInterpolator(
        g, np.array([-1.0, 1.0, 2.5, 5.0, 10.0]) * u.Hz, np.float64
    )
    sky = jnp.broadcast_to(
        jnp.array([-2.0, 4.0, 14.0])[None, None, :, None, None], g.shape
    )
    # Expected sky values for interpolation with contribution weights for centers [1, 4, 9]:
    # - frequency -1 : Outside of grid range/ left to the first bin center; interpolation assigns nearest bin center value; expected: -2; weights: [1,0,0]
    # - frequency 1 : Exactly at first bin center; interpolation assigns that bin center value; expected : -2; weights: [1,0,0]
    # - frequency 2.5 : Halfway between first and second center; Interpolation assigns equal weights; expected: 0.5*(-2+4) = 1; weights: [0.5,0.5,0]
    # - frequency 5 : Between centers 4 and 9, closer to 4; expected: 0.8*4 + 0.2*14 = 6; weights: [0,0.8,0.2]
    # - frequency 10 : Beyond the last center (9); nearest endpoint brightness is 14; weights: [0,0,1]
    expected_interpolated_sky = [-2, -2, 1, 6, 14]
    np.testing.assert_allclose(
        jax.jit(op)(sky)[0, 0, :, 0, 0], expected_interpolated_sky
    )
    # To test gradients a scalar output is needed, thus summation as gradient is a linear operation
    # Gradient ask for contriubtion to original sky frequency. thus the output for every interpolated target frequency is the weights array from above
    # Expected output is sum off all weights arrays because of the summation operation.
    expected_gradient = [2.5, 1.3, 1.2]
    gradient = jax.grad(lambda x: jnp.sum(op(x)))(sky)
    np.testing.assert_allclose(gradient[0, 0, :, 0, 0], expected_gradient)


def test_gaps_nearest_and_ties():
    # Tests gap mode nearest
    # Continous spectral intervals are from 0 to 4 and 8 to 12
    g = grid(np.array([[0.0, 2.0], [2.0, 4.0], [8.0, 10.0], [10.0, 12.0]]))

    op = SpectralInterpolator(
        g, np.array([2.0, 5.0, 6.0, 7.0, 10.0]) * u.Hz, np.float64
    )
    sky = jnp.broadcast_to(
        jnp.array([1.0, 3.0, 9.0, 11.0])[None, None, :, None, None], g.shape
    )
    # Expected sky values with contribution weights for centers [1, 3, 9, 11]:
    # - frequency 2: Halfway between centers 1 and 3 within the first range;
    #   expected: 0.5*1 + 0.5*3 = 2; weights: [0.5, 0.5, 0, 0]
    # - frequency 5: In the gap, nearest center is 3;
    #   expected: 3; weights: [0, 1, 0, 0]
    # - frequency 6: In the gap, equally distant from centers 3 and 9;
    #   the tie selects the lower center; expected: 3; weights: [0, 1, 0, 0]
    # - frequency 7: In the gap, nearest center is 9;
    #   expected: 9; weights: [0, 0, 1, 0]
    # - frequency 10: Halfway between centers 9 and 11 within the second range;
    #   expected: 0.5*9 + 0.5*11 = 10; weights: [0, 0, 0.5, 0.5]
    expected_sky = [2, 3, 3, 9, 10]
    np.testing.assert_allclose(op(sky)[0, 0, :, 0, 0], expected_sky)


def test_single_unbounded_bin_complex():
    # If single frequency sky interpolation assigns that value to all target frequencies
    g = grid(np.array([-np.inf, np.inf]))
    op = SpectralInterpolator(g, np.array([1.0, 2.0, 3.0]) * u.Hz, np.complex128)
    sky = jnp.full(g.shape, -2 + 3j)
    np.testing.assert_allclose(op(sky), -2 + 3j)
    assert op.target.shape == (1, 1, 3, 2, 3)


def test_beam_and_response_use_exact_data_channels(monkeypatch):
    from generate_test_obs import generate_random_obs
    from jubik.polarization import PolarizationType
    from jubik.instruments.resolve.mosaicing.sky_beamer import build_sky_beamer
    from jubik.instruments.resolve.parse.response import FinufftSettings
    import importlib

    response_module = importlib.import_module("jubik.instruments.resolve.response")
    frequencies = np.array([1.5, 2.5, 3.5])
    obs = generate_random_obs(
        frequencies,
        4,
        [-1, 1],
        [-1, 1],
        PolarizationType(("LL", "RR")),
        fraction_flagged=0.0,
    )
    g = grid(np.array([0.0, 2.0, 4.0]))
    op = SpectralInterpolator(g, frequencies * u.Hz, np.float64)
    sky = jnp.broadcast_to(jnp.array([1.0, 3.0])[None, None, :, None, None], g.shape)
    seen = []

    # Beam (same shape as full sky) per data frequency sets frequency as the beam value; beam has sam
    def beam(freq, x):
        seen.append(freq)
        return np.full_like(x, freq)

    from types import SimpleNamespace
    from jubik.instruments.resolve.data.direction import Direction

    direction = Direction(np.array([0.0, 0.0]), 2000)
    pointing = SimpleNamespace(
        direction=direction, direction_from_key=lambda key: direction
    )
    beamer = build_sky_beamer(
        op.target, g.spatial, op.target_frequencies, [pointing], beam
    )

    def backend(observation, **kwargs):
        return lambda image: jnp.full(
            (observation.nrow, observation.nfreq), image.mean()
        )

    monkeypatch.setattr(response_module, "interferometry_response_finufft", backend)
    response = response_module.interferometry_response(
        obs, g, FinufftSettings(1e-6), channel_frequencies=frequencies
    )
    # Expected interpolated brightness for source centers [1, 3] and values [1, 3]:
    # - frequency 1.5: w = (1.5 - 1)/(3 - 1) = 0.25;
    #   expected: 0.75*1 + 0.25*3 = 1.5; contribution weights: [0.75, 0.25]
    # - frequency 2.5: w = (2.5 - 1)/(3 - 1) = 0.75;
    #   expected: 0.25*1 + 0.75*3 = 2.5; contribution weights: [0.25, 0.75]
    # - frequency 3.5: w = (3.5 - 1)/(3 - 1) = 1.25, clipped to 1;
    #   expected: 0*1 + 1*3 = 3; contribution weights: [0, 1]
    # The test beam multiplies each slice by its data frequency:
    # [1.5*1.5, 2.5*2.5, 3*3.5] = [2.25, 6.25, 10.5].
    beamed = next(iter(beamer(op(sky)).values()))
    np.testing.assert_allclose(seen, frequencies)
    np.testing.assert_allclose(
        response(beamed), np.broadcast_to([2.25, 6.25, 10.5], obs.vis.shape)
    )
    # Tests if rejection when channel_frequencies and observation frequencies differ in value
    with pytest.raises(ValueError, match="channel frequencies"):
        response_module.interferometry_response(
            obs, g, FinufftSettings(1e-6), channel_frequencies=frequencies + 1
        )


def test_radio_extractor_slices_normalized_frequency_axis():
    import nifty.re as jft
    from jubik.instruments.resolve.multimessanger.radio_sky_extractor import (
        build_radio_sky_extractor,
    )

    for shape in [(3, 2, 3), (2, 1, 3, 2, 3)]:
        extractor = build_radio_sky_extractor(1, jft.ShapeWithDtype(shape, np.float64))
        result = extractor(jnp.arange(np.prod(shape)).reshape(shape))
        assert result.shape == ((1, 1, 2, 2, 3) if len(shape) == 3 else (2, 1, 2, 2, 3))


@pytest.mark.parametrize(
    "gap_mode, expected",
    [
        ("nearest", [1, 3, 3, 9, 11]),
        ("linear", [1, 5, 6, 7, 11]),
    ],
)
def test_configurable_gap_mode_preserves_outer_endpoints(gap_mode, expected):
    g = grid(np.array([[0.0, 2.0], [2.0, 4.0], [8.0, 10.0], [10.0, 12.0]]))
    op = SpectralInterpolator(
        g, np.array([-1.0, 5.0, 6.0, 7.0, 13.0]) * u.Hz, np.float64, gap_mode=gap_mode
    )
    sky = jnp.broadcast_to(
        jnp.array([1.0, 3.0, 9.0, 11.0])[None, None, :, None, None], g.shape
    )
    # Expected sky values for centers and brightness [1, 3, 9, 11]:
    # - frequency -1: Outside the grid to the left; both modes return 1;
    #   contribution weights: [1, 0, 0, 0].
    # - frequency 5: In the gap;
    #   nearest: expected: 3, weights: [0, 1, 0, 0].
    #   Linear: w = (5 - 3)/(9 - 3) = 1/3; expected: (2/3)*3 + (1/3)*9 = 5; weights: [0, 2/3, 1/3, 0].
    # - frequency 6: In the gap;
    #   nearest:  tie selects 3, weights: [0, 1, 0, 0].
    #   Linear: w = (6 - 3)/(9 - 3) = 0.5; expected: 0.5*3 + 0.5*9 = 6; weights: [0, 0.5, 0.5, 0].
    # - frequency 7: In the gap;
    #   nearest: expected: 9, weights: [0, 0, 1, 0].
    #   Linear: w = (7 - 3)/(9 - 3) = 2/3;
    #   expected: (1/3)*3 + (2/3)*9 = 7; weights: [0, 1/3, 2/3, 0].
    # - frequency 13: Outside the grid to the right; both modes return 11;
    #   contribution weights: [0, 0, 0, 1].
    np.testing.assert_allclose(jax.jit(op)(sky)[0, 0, :, 0, 0], expected)


def test_invalid_gap_mode():
    with pytest.raises(ValueError, match="gap_mode"):
        SpectralInterpolator(
            grid(np.array([0.0, 2.0])),
            np.array([1.0]) * u.Hz,
            np.float64,
            gap_mode="invalid",
        )


@pytest.mark.parametrize("enabled", [None, False, True])
def test_radio_likelihood_selects_complete_pipeline(monkeypatch, enabled):
    import importlib
    from types import SimpleNamespace
    import nifty.re as jft
    from jubik.instruments.resolve.data.direction import Direction

    module = importlib.import_module(
        "jubik.instruments.resolve.likelihood.radio_likelihood"
    )
    g = grid(np.array([0.0, 2.0, 4.0]))
    direction = Direction(np.array([0.0, 0.0]), 2000)
    observations = [
        SimpleNamespace(
            freq=np.array([1.5, 2.5]),
            direction=direction,
            direction_from_key=lambda key: direction,
        ),
        SimpleNamespace(
            freq=np.array([2.0]),
            direction=direction,
            direction_from_key=lambda key: direction,
        ),
    ]
    monkeypatch.setattr(
        module, "DataLoading", SimpleNamespace(from_yaml_dict=lambda cfg: None)
    )
    monkeypatch.setattr(
        module, "ObservationModify", SimpleNamespace(from_yaml_dict=lambda cfg: None)
    )
    monkeypatch.setattr(
        module, "BeamPatternConfig", SimpleNamespace(from_yaml_dict=lambda cfg: None)
    )
    monkeypatch.setattr(
        module, "load_and_modify_data_from_objects", lambda **kwargs: observations
    )
    # Sets beam function to be 1 everywhere
    monkeypatch.setattr(
        module,
        "build_primary_beam_pattern_from_beam_pattern_config",
        lambda cfg: lambda freq, x: np.ones_like(x),
    )
    monkeypatch.setattr(module, "factory_noise_correction_parser", lambda cfg: None)
    calls = []

    def likelihood_builder(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module, "build_likelihood_from_sky_beamer", likelihood_builder)
    cfg = {
        "radio_response": {"backend": "finufft", "epsilon": 1e-6},
        "alma_data": {"demo": {"dish": {}}},
    }
    if enabled is not None:
        cfg["radio_spectral_interpolation"] = {"enabled": enabled, "gap_mode": "linear"}
    products = module.build_radio_likelihood(
        ["demo"], cfg, g, jft.ShapeWithDtype((2, 2, 3), np.float64), None
    )
    # Field values are chosen such that when interpolating to data frequencies the interpolated values are equal to the data frequencies
    # Beam is set to 1 everywhere so beamed sky has the same entires as the sky before applying the beam
    sky_values_per_freq = [1.0, 3.0]
    # Builds a flat sky per sky frequency
    sky = jnp.broadcast_to(jnp.array(sky_values_per_freq)[:, None, None], (2, 2, 3))
    result = products.sky_beamer(sky)
    assert len(calls) == 2
    if enabled:
        assert len(result) == 2  # 2 entries because of one beamer per observation
        assert (
            calls[0]["sky_beamer"] is not calls[1]["sky_beamer"]
        )  # Because of different data frequencies beamers have to be different
        for call, obs in zip(calls, observations):
            # Checks if direct channel mapping was requested in response
            np.testing.assert_array_equal(call["channel_frequencies"], obs.freq)
            # Verifies that pipeline produces the expected beamed sky interpolated
            np.testing.assert_allclose(
                result[call["field_name"]][0, 0, :, 0, 0], obs.freq
            )
    else:
        assert (
            len(result) == 1
        )  # 1 entry because beamer is reused since the pointing of both observations is the same
        assert (
            calls[0]["sky_beamer"] is calls[1]["sky_beamer"]
        )  # Has to be the same as beamer is reused
        assert (
            calls[0]["field_name"] == calls[1]["field_name"]
        )  # Checks for same beamed sky as beamer is reused for both observations
        assert all(
            "channel_frequencies" not in call for call in calls
        )  # Checks that direct sky bins to channel mapping was not requested
        # Verifies that pipeline produces the expected beamed sky
        # As beamer (which is 1 everywhere) is applied to the non interpolated sky the values do not change
        np.testing.assert_allclose(
            next(iter(result.values()))[0, 0, :, 0, 0], sky_values_per_freq
        )
