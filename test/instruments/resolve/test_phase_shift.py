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
from jubik.instruments.resolve.parse.phase_shift import (
    ShiftGroup,
    assign_shift_groups,
    parse_shift_groups,
)
from jubik.instruments.resolve.phase_shift import PhaseShift, build_shift_models
from jubik.instruments.resolve.response import _hermitian_adjoint

from generate_test_obs import generate_random_obs
from test_response import setup_grid_obs_response

jax.config.update("jax_enable_x64", True)

FREQS = np.array([1e9, 1.3e9])


def shift_model(prior):
    """The shift of one group `eb1`, domain key `shift_eb1`."""
    return build_shift_models([ShiftGroup("eb1", prior, [])])["eb1"]


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
    shift = jft.Model(lambda x: x["s"], domain={"s": jft.ShapeWithDtype((2,))})

    phase = PhaseShift(shift, obs)({"s": np.array([a * pix_row, b * pix_col])})

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
    shift = shift_model(["normal", 0.0, 0.05])

    phase = PhaseShift(shift, obs)(jft.zeros_like(shift.domain))

    assert phase.dtype == vis_dtype
    assert phase.shape == (1,) + obs.vis.shape[1:]
    assert_allclose(phase, 1.0)


def data_cfg(templates, windows):
    return {
        "data_path": "/data",
        "data_templates": templates,
        "spectral": {"window": windows},
    }


def test_template_covers_every_spw_and_blocks_share_a_group():
    groups = parse_shift_groups(
        {
            "eb1": {"prior": ["normal", 0.0, 0.05], "data": ["a/a_spw{spw}.npz"]},
            "eb2_usb": {"prior": ["normal", 0.0, 0.1], "data": ["b/b_spw17.npz"]},
        }
    )
    templates = ["a/a_spw{spw}.npz", "b/b_spw{spw}.npz", "c/c_spw{spw}.npz"]

    assigned = assign_shift_groups(
        groups,
        {"lsb": data_cfg(templates, [5, 7]), "usb": data_cfg(templates, [17])},
    )

    # loader_templates order: template, then spw; c is in no group
    assert assigned["lsb"] == ["eb1", "eb1", None, None, None, None]
    assert assigned["usb"] == ["eb1", "eb2_usb", None]
    assert parse_shift_groups(None) == []


def test_file_in_two_groups_raises():
    groups = parse_shift_groups(
        {
            "eb1": {"prior": ["normal", 0.0, 0.05], "data": ["a/a_spw{spw}.npz"]},
            "eb1_usb": {"prior": ["normal", 0.0, 0.1], "data": ["a/a_spw17.npz"]},
        }
    )
    with pytest.raises(ValueError, match="a_spw17.npz"):
        assign_shift_groups(groups, {"b": data_cfg(["a/a_spw{spw}.npz"], [5, 17])})


def test_entry_of_an_unloaded_block_is_ignored():
    groups = parse_shift_groups(
        {"eb1": {"prior": ["normal", 0.0, 0.05], "data": ["a/a_spw9.npz"]}}
    )
    assigned = assign_shift_groups(groups, {"b": data_cfg(["a/a_spw{spw}.npz"], [5])})
    assert assigned == {"b": [None]}


def test_equivalent_paths_match():
    groups = parse_shift_groups(
        {"eb1": {"prior": ["normal", 0.0, 0.05], "data": ["a/a_spw5.npz"]}}
    )
    cfg = data_cfg(["./a/a_spw{spw}.npz"], [5])
    cfg["data_path"] = "/data/"
    assert assign_shift_groups(groups, {"b": cfg}) == {"b": ["eb1"]}


def test_group_prior_sets_the_shift():
    assert_allclose(
        shift_model(["normal", 0.0, 0.02])({"shift_eb1": np.ones(2)}), 0.02
    )


def test_likelihood_has_the_shift_only_when_set():
    np.random.seed(44)
    grid, obs, _, _ = setup_grid_obs_response(("I",), ("LL", "RR"), FREQS)
    sky_beamer = SkyBeamer(
        jft.ShapeWithDtype(grid.shape),
        {"band7_fld": BeamPattern(0.0, 0.0, np.ones(grid.shape), None)},
    )

    for prior in (None, ["normal", 0.0, 0.05]):
        shift = None if prior is None else shift_model(prior)
        likelihood = build_likelihood_from_sky_beamer(
            observation=obs,
            field_name="band7_fld",
            sky_beamer=sky_beamer,
            sky_grid=grid,
            backend_settings=rve.parse.FinufftSettings(epsilon=1e-9),
            shift=None if shift is None else PhaseShift(shift, obs),
        ).likelihood
        domain = likelihood.domain.tree

        assert ("shift_eb1" in domain) == (prior is not None)
        latent = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
        assert np.isfinite(likelihood.energy(latent))

    assert domain["shift_eb1"].shape == (2,)


def build_band7(backend, prior):
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
    shift = None if prior is None else shift_model(prior)
    lhb = build_likelihood_from_sky_beamer(
        observation=obs,
        field_name="band7_fld",
        sky_beamer=sky_beamer,
        sky_grid=grid,
        backend_settings=settings,
        shift=None if shift is None else PhaseShift(shift, obs),
    )
    return lhb, {"band7_fld": np.random.normal(size=grid.shape)}


@pytest.mark.parametrize("backend", ("ducc", "finufft"))
def test_response_adjoint_holds_the_shift_fixed(backend):
    """<R_x s, d> == <s, R_x^H d> with the shift fixed at a nonzero sample x."""
    # 60 arcsec std: phases of order one on the short test baselines
    lhb, sky = build_band7(backend, ["normal", 0.0, 60.0])
    unshifted, _ = build_band7(backend, None)
    x = {"shift_eb1": np.array([0.7, -1.3])}
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
