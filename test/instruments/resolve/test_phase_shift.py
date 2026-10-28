import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.likelihood.mosaic_likelihood import (
    build_likelihood_from_sky_beamer,
)
from jubik.instruments.resolve.mosaicing.sky_beamer import BeamPattern, SkyBeamer
from jubik.instruments.resolve.phase_shift import PhaseShift, build_shift_prior
from jubik.instruments.resolve.response import _hermitian_adjoint

from generate_test_obs import generate_random_obs
from test_response import setup_grid_obs_response

jax.config.update("jax_enable_x64", True)

FREQS = np.array([1e9, 1.3e9])


def point_source_sky(grid, i, j):
    sky = np.zeros(grid.shape)
    sky[..., i, j] = 1.0
    return sky


@pytest.mark.parametrize("backend", ("ducc", "finufft"))
def test_shift_moves_the_sampling_positions(backend):
    """A shift of (a, b) pixels along the sky grid (row, column) moves the
    sampling positions of the model sky by (a, b), so a source at pixel
    (i, j) gives the visibilities of a source at (i - a, j - b) without shift.
    """
    np.random.seed(42)
    grid, obs, r_ducc, r_finufft = setup_grid_obs_response(("I",), ("LL", "RR"), FREQS)
    response = r_ducc if backend == "ducc" else r_finufft
    a, b, i, j = 3, -2, 40, 55
    pix_row, pix_col = grid.spatial.pixel_scales_yx.to_value(u.arcsec)
    # Row 1 belongs to the observation, row 0 must not be read.
    shifts = np.array([[1.0, 1.0], [a * pix_row, b * pix_col]])
    shift = jft.Model(lambda x: x["s"], domain={"s": jft.ShapeWithDtype((2, 2))})

    phase = PhaseShift(shift, 1, obs)({"s": shifts})

    assert_allclose(
        phase * response(point_source_sky(grid, i, j)),
        response(point_source_sky(grid, i - a, j - b)),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "vis_dtype,weight_dtype", ((np.complex64, np.float32), (np.complex128, np.float64))
)
def test_zero_shift_is_identity(vis_dtype, weight_dtype):
    np.random.seed(43)
    pol = ju.polarization.PolarizationType(("LL", "RR"))
    obs = generate_random_obs(FREQS, 10, [-1e2, 1e2], [-5, 5], pol)
    obs = rve.Observation(
        obs.antenna_positions,
        obs.vis_val.astype(vis_dtype),
        obs.weight_val.astype(weight_dtype),
        obs.legacy_polarization,
        obs.freq,
        None,
    )
    shift = build_shift_prior("band7", ["normal", 0.0, 0.05], 3)

    phase = PhaseShift(shift, 2, obs)(jft.zeros_like(shift.domain))

    assert phase.dtype == vis_dtype
    assert phase.shape == (1,) + obs.vis.shape[1:]
    assert_allclose(phase, 1.0)


def test_band_entry_wins_over_default():
    settings = {"default": ["normal", 0.0, 0.05], "band7": ["normal", 0.0, 0.02]}
    one_sigma = lambda band, s: build_shift_prior(band, s, 2)(
        {f"{band}_shift": np.ones((2, 2))}
    )

    assert_allclose(one_sigma("band7", settings), 0.02)
    assert_allclose(one_sigma("band6", settings), 0.05)
    assert_allclose(one_sigma("band6", ["normal", 0.0, 0.05]), 0.05)
    assert build_shift_prior("band7", None, 2) is None


def test_likelihood_has_band_shift_only_when_set():
    np.random.seed(44)
    grid, obs, _, _ = setup_grid_obs_response(("I",), ("LL", "RR"), FREQS)
    sky_beamer = SkyBeamer(
        jft.ShapeWithDtype(grid.shape),
        {"band7_fld": BeamPattern(0.0, 0.0, np.ones(grid.shape), None)},
    )

    for shift_settings in (None, {"default": ["normal", 0.0, 0.05]}):
        shift = build_shift_prior("band7", shift_settings, 2)
        likelihood = build_likelihood_from_sky_beamer(
            observation=obs,
            field_name="band7_fld",
            sky_beamer=sky_beamer,
            sky_grid=grid,
            backend_settings=rve.parse.FinufftSettings(epsilon=1e-9),
            shift=None if shift is None else PhaseShift(shift, 1, obs),
        ).likelihood
        domain = likelihood.domain.tree

        assert ("band7_shift" in domain) == (shift_settings is not None)
        latent = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
        assert np.isfinite(likelihood.energy(latent))

    assert domain["band7_shift"].shape == (2, 2)


def build_band7(backend, shift_settings):
    """Builder of one observation of band7 with one field, and a sky sample."""
    np.random.seed(45)
    grid, obs, _, _ = setup_grid_obs_response(("I",), ("LL", "RR"), FREQS)
    sky_beamer = SkyBeamer(
        jft.ShapeWithDtype(grid.shape),
        {"band7_fld": BeamPattern(0.0, 0.0, np.ones(grid.shape), None)},
    )
    settings = {
        "ducc": rve.parse.Ducc0Settings(
            epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False
        ),
        "finufft": rve.parse.FinufftSettings(epsilon=1e-9),
    }[backend]
    shift = build_shift_prior("band7", shift_settings, 2)
    lhb = build_likelihood_from_sky_beamer(
        observation=obs,
        field_name="band7_fld",
        sky_beamer=sky_beamer,
        sky_grid=grid,
        backend_settings=settings,
        shift=None if shift is None else PhaseShift(shift, 1, obs),
    )
    return lhb, {"band7_fld": np.random.normal(size=grid.shape)}


@pytest.mark.parametrize("backend", ("ducc", "finufft"))
def test_response_adjoint_holds_the_shift_fixed(backend):
    """<R_x s, d> == <s, R_x^H d> with the shift fixed at a nonzero sample x."""
    # 60 arcsec std: phases of order one on the short test baselines
    lhb, sky = build_band7(backend, ["normal", 0.0, 60.0])
    unshifted, _ = build_band7(backend, None)
    x = {"band7_shift": np.array([[0.0, 0.0], [0.7, -1.3]])}
    d = lhb.visibilities

    adjoint = lhb.response_adjoint(d, x)["band7_fld"]

    lhs = np.real(np.sum(np.conj(lhb.response(sky | x)) * d))
    assert_allclose(lhs, np.sum(sky["band7_fld"] * np.real(adjoint)), rtol=1e-6)
    # The phase reaches the adjoint, so it differs from the unshifted one.
    plain = unshifted.response_adjoint(d)["band7_fld"]
    assert np.linalg.norm(adjoint - plain) > 0.1 * np.linalg.norm(plain)
    assert_allclose(lhb.response_adjoint(d, jft.Vector(x))["band7_fld"], adjoint)


def test_response_adjoint_needs_the_sample_with_a_shift():
    lhb, _ = build_band7("finufft", ["normal", 0.0, 1.0])
    with pytest.raises(ValueError, match="shift"):
        lhb.response_adjoint(lhb.visibilities)


def test_response_adjoint_without_shift_is_unchanged():
    lhb, _ = build_band7("finufft", None)
    d = lhb.visibilities
    expected = _hermitian_adjoint(lhb.response, lhb.response.domain, d)
    assert_allclose(
        lhb.response_adjoint(d, {"ignored": 0})["band7_fld"], expected["band7_fld"]
    )
