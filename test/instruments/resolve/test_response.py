import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.response import (
    interferometry_response_ducc,
    interferometry_response_finufft,
)

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


def setup_grid_obs_response(pol_sky, pol_channels, freqs):
    pol_type_sky = ju.polarization.PolarizationType(pol_sky)
    pol_type_data = ju.polarization.PolarizationType(pol_channels)
    obs = generate_random_obs(
        freqs, 50, [-1e2, 1e2], [-5, 5], pol_type_data, fraction_flagged=0.3,
        rng=np.random.default_rng(42),
    )
    sky_center = SkyCoord(ra=np.nan * u.Unit("rad"), dec=np.nan * u.Unit("rad"))
    fov = u.Quantity((u.Quantity("1deg"), u.Quantity("1.5deg")))
    spacial = ju.wcs.WcsAstropy(center=sky_center, shape=(100, 100), fov=fov)
    spectral = ju.color.Color.from_central_frequencies(freqs)
    grid = ju.Grid(spatial=spacial, spectral=spectral, polarization=pol_type_sky)

    ducc0_settings = rve.parse.Ducc0Settings(
        epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False
    )
    finufft_settings = rve.parse.FinufftSettings(epsilon=1e-9)
    r_ducc = rve.interferometry_response(obs, grid, backend_settings=ducc0_settings)
    r_finufft = rve.interferometry_response(
        obs, grid, backend_settings=finufft_settings
    )
    return grid, obs, r_ducc, r_finufft


@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
@pmp(
    "center_pixels",
    ((0.0, 0.0), (3.0, 0.0), (0.0, -2.0), (3.0, -2.0)),
    ids=("unshifted", "x-shift", "y-shift", "xy-shift"),
)
def test_ducc_finufft_backend_consistency(freqs, center_pixels):
    rng = np.random.default_rng(42)

    fov = (u.Quantity("1deg"), u.Quantity("1.5deg"))

    npix_x, npix_y = 32, 40
    pixsize_x = fov[0].to_value(u.radian) / npix_x
    pixsize_y = fov[1].to_value(u.radian) / npix_y
    # Approach Nyquist at the highest frequency while staying inside both axes.
    uv_limit = 0.98 * c.value / (2 * max(pixsize_x, pixsize_y) * max(freqs))

    pol = ju.polarization.PolarizationType(("I",))
    obs = generate_random_obs(
        freqs=freqs,
        n_rows=20,
        uv_range=[-uv_limit, uv_limit],
        # FINUFFT requires coplanar arrays thus w=0
        w_range=[0.0, 0.0],
        polarization_type=pol,
        rng=rng,
    )

    # Leave margin between kernel accuracy and the comparison tolerance.
    epsilon = 1e-12
    # Separate axis shifts expose phase-sign errors and swapped center axes.
    center_x = center_pixels[0] * pixsize_x
    center_y = center_pixels[1] * pixsize_y

    response_ducc = interferometry_response_ducc(
        observation=obs,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        # No wgridding as a coplanar array is assumed
        do_wgridding=False,
        epsilon=epsilon,
        nthreads=1,
        verbosity=False,
        center_x=center_x,
        center_y=center_y,
    )
    response_finufft = interferometry_response_finufft(
        observation=obs,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        center_x=center_x,
        center_y=center_y,
    )

    image = jnp.asarray(
        rng.normal(size=(npix_x, npix_y)),
        dtype=jnp.float64,
    )

    vis_ducc = jax.jit(response_ducc)(image)
    vis_finufft = jax.jit(response_finufft)(image)
    assert_allclose(vis_ducc, response_ducc(image), rtol=1e-12, atol=1e-15)
    assert_allclose(vis_finufft, response_finufft(image), rtol=1e-12, atol=1e-15)

    assert vis_ducc.shape == obs.vis.shape[1:]
    assert vis_finufft.shape == vis_ducc.shape
    # Remove the pixel-area factor so atol measures transform accuracy.
    vol = pixsize_x * pixsize_y
    assert_allclose(vis_ducc / vol, vis_finufft / vol, rtol=1e-9, atol=1e-9)

    # The sky is real: its adjoint uses the real part of the complex pairing.
    y = jnp.asarray(rng.normal(size=vis_ducc.shape) + 1j * rng.normal(size=vis_ducc.shape))
    gradients = []
    for response in (response_ducc, response_finufft):
        normalized = lambda x: response(x) / vol
        values, pullback = jax.vjp(normalized, image)
        adjoint_y = pullback(jnp.conj(y))[0]
        assert_allclose(
            jnp.real(jnp.vdot(values, y)), jnp.vdot(image, adjoint_y),
            rtol=1e-10, atol=1e-9,
        )
        loss = lambda x: 0.5 * jnp.sum(jnp.abs(normalized(x) - y) ** 2)
        gradients.append(jax.jit(jax.grad(loss))(image))
    assert_allclose(gradients[0], gradients[1], rtol=1e-9, atol=1e-7)


@pmp(
    "center_kwargs",
    ({}, {"center_x": None, "center_y": None},
     {"center_x": None, "center_y": 0.0},
     {"center_x": 0.0, "center_y": None},
     {"center_x": None, "center_y": 2e-4},
     {"center_x": 1e-4, "center_y": None}),
    ids=("defaults", "both-none", "x-none-y-zero", "x-zero-y-none",
         "x-none-y-shift", "x-shift-y-none"),
)
def test_finufft_missing_phase_center_skips_shift(center_kwargs):
    rng = np.random.default_rng(42)
    obs = generate_random_obs(
        np.array([1e9, 1.3e9]), 20, [-1e2, 1e2], [0.0, 0.0],
        ju.polarization.PolarizationType(("I",)), rng=rng,
    )
    kwargs = dict(observation=obs, pixsize_x=1e-4, pixsize_y=2e-4, epsilon=1e-12)
    default_response = interferometry_response_finufft(**kwargs, **center_kwargs)
    zero_response = interferometry_response_finufft(
        **kwargs, center_x=0.0, center_y=0.0
    )
    image = rng.normal(size=(32, 40))

    assert_allclose(default_response(image), zero_response(image), rtol=0, atol=0)


@pmp("pol_sky", (("I",), ("I", "Q", "U", "V")))
@pmp("pol_channels", (("LL", "RR"), ("RR", "RL", "LR", "LL"), ("XX", "XY", "YX", "YY")))
@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
def test_response_ducc_finufft_consistency(pol_sky, pol_channels, freqs):
    if pol_sky == ("I", "Q", "U", "V") and pol_channels == ("LL", "RR"):
        pytest.skip("Only stokes I data.")
    if pol_sky == ("I",) and not pol_channels == ("LL", "RR"):
        pytest.skip("Only stokes I sky.")
    grid, obs, r_ducc, r_finufft = setup_grid_obs_response(pol_sky, pol_channels, freqs)

    sky = np.random.default_rng(42).normal(size=grid.shape)
    res_ducc = r_ducc(sky)
    res_finufft = r_finufft(sky)

    assert_allclose(res_ducc, res_finufft, rtol=1e-9, atol=1e-9)


@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
def test_response_StokesI(freqs):
    pol_sky = ("I",)
    pol_channels = ("LL", "RR")
    grid, obs, r_ducc, r_finufft = setup_grid_obs_response(pol_sky, pol_channels, freqs)

    sky = jnp.zeros(grid.shape)
    cx = grid.shape[3] // 2
    cy = grid.shape[4] // 2
    dvol = grid.spatial.dvol.to(u.rad**2).value
    sky[:, :, :, cx, cy] = 1 / dvol
    vis_ducc = r_ducc(sky)
    vis_finufft = r_finufft(sky)
    vis_expected = np.ones(obs.vis.shape, dtype=jnp.complex128)
    assert_allclose(vis_ducc, vis_expected)
    assert_allclose(vis_finufft, vis_expected)


def stokes_to_circular(iquv):
    i, q, u, v = iquv
    return jnp.array([i + v, q + 1j * u, q - 1j * u, i - v])


def stokes_to_linear(iquv):
    i, q, u, v = iquv
    return jnp.array([i + q, u + 1j * v, u - 1j * v, i - q])


@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
@pmp("circular", (True, False))
@pmp(
    "source",
    (
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.3, -0.4, 0.0),
        (1.0, 0.0, 0.0, -0.6),
        (1.0, 0.36, -0.48, 0.8),
    ),
    ids=("unpolarized", "linear", "circular", "fully-polarized"),
)
def test_response_StokesIQUV(freqs, circular, source):
    pol_sky = ("I", "Q", "U", "V")
    # Physical constraints are covered in test/sky_model/test_stokes_adder.py;
    # these explicit sources exercise the linear polarization conversions.
    source = np.asarray(source)

    if circular:
        pol_channels = ("RR", "RL", "LR", "LL")
    else:
        pol_channels = ("XX", "XY", "YX", "YY")
    grid, obs, r_ducc, r_finufft = setup_grid_obs_response(pol_sky, pol_channels, freqs)

    sky = jnp.zeros(grid.shape)
    cx = grid.shape[3] // 2
    cy = grid.shape[4] // 2
    dvol = grid.spatial.dvol.to(u.rad**2).value
    sky[:, 0, :, cx, cy] = source[np.newaxis].T / dvol
    vis_ducc = r_ducc(sky)
    vis_finufft = r_finufft(sky)
    if circular:
        res = stokes_to_circular(source)
    else:
        res = stokes_to_linear(source)
    res = res[:, np.newaxis, np.newaxis]
    vis_expected = np.empty(obs.vis.shape, dtype=np.complex128)
    vis_expected[:, :, :] = res
    assert_allclose(vis_ducc, vis_expected)
    assert_allclose(vis_finufft, vis_expected)
