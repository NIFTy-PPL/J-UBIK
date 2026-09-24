"""cufinufft response against finufft, ducc0 and analytic point sources. GPU only.

Lives apart from test_response.py until the rewrite of that file (MR 232)
lands; then move these tests over.
"""

import importlib.util

import jax
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.data.auxiliary_table import AuxiliaryTable
from jubik.instruments.resolve.parse.response import CufinufftSettings
from jubik.instruments.resolve.response import (
    CufinufftResponse,
    interferometry_response_ducc,
    interferometry_response_finufft,
)

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

CUFINUFFT_SETTINGS = CufinufftSettings(epsilon=1e-10, gpu_maxbatchsize=0, upsampfac=2.0)
EPSILON = 1e-10
TOLERANCE = 1e-8
SKY_CENTER = (1.0, 0.5)  # ra, dec in rad


@pytest.fixture(scope="module", autouse=True)
def cuda_backend():
    try:
        jax.devices("cuda")
    except RuntimeError:
        pytest.skip("CUDA-enabled JAX is required")
    for module in (
        "cufinufft",
        "jax_finufft",
        "jubik.instruments.resolve.cufinufft._exec",
    ):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed/built")
    with jax.enable_x64(True):
        yield
    jax.clear_caches()


def build_responses(pol_sky, pol_channels, freqs, phase_center_offset=None):
    pol_type_data = ju.polarization.PolarizationType(pol_channels)
    obs = generate_random_obs(freqs, 50, [-1e2, 1e2], [-5, 5], pol_type_data)
    if phase_center_offset is None:
        sky_center = SkyCoord(ra=np.nan * u.rad, dec=np.nan * u.rad)
    else:
        sky_center = SkyCoord(*(SKY_CENTER * u.rad), frame="icrs")
        phase_center = np.add(SKY_CENTER, phase_center_offset).reshape(1, 1, 2)
        field = AuxiliaryTable({"REFERENCE_DIR": phase_center})
        obs = rve.Observation(
            obs.antenna_positions,
            obs.vis_val,
            obs.weight_val,
            obs.legacy_polarization,
            obs.freq,
            {"FIELD": field},
        )
    fov = u.Quantity((u.Quantity("0.5deg"), u.Quantity("0.75deg")))
    spatial = ju.wcs.WcsAstropy(center=sky_center, shape=(16, 24), fov=fov)
    spectral = ju.color.Color.from_central_frequencies(freqs)
    polarization = ju.polarization.PolarizationType(pol_sky)
    grid = ju.Grid(spatial=spatial, spectral=spectral, polarization=polarization)

    # The reference runs on CPU so the comparison does not depend on how
    # jax_finufft was built.
    with jax.default_device(jax.devices("cpu")[0]):
        r_finufft = rve.interferometry_response(
            obs, grid, backend_settings=rve.parse.FinufftSettings(epsilon=EPSILON)
        )
    with jax.default_device(jax.devices("cuda")[0]):
        r_cufinufft = rve.interferometry_response(
            obs,
            grid,
            backend_settings=CUFINUFFT_SETTINGS,
        )
    return grid, obs, r_finufft, r_cufinufft


def assert_transform(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert_allclose(
        actual,
        expected,
        rtol=TOLERANCE,
        atol=TOLERANCE * np.max(np.abs(expected)),
    )


def on_device(x, platform):
    return jax.device_put(x, jax.devices(platform)[0])


@pmp(
    "pol_sky,pol_channels",
    ((("I",), ("LL", "RR")), (("I", "Q", "U", "V"), ("XX", "XY", "YX", "YY"))),
)
@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
@pmp("phase_center_offset", (None, (1e-3, -2e-3)))
def test_forward_and_vjp_match_finufft(
    pol_sky, pol_channels, freqs, phase_center_offset
):
    np.random.seed(42)
    grid, obs, r_finufft, r_cufinufft = build_responses(
        pol_sky, pol_channels, freqs, phase_center_offset
    )
    sky = np.random.normal(size=grid.shape)
    cotangent = np.random.normal(size=obs.vis.shape) + 1j * np.random.normal(
        size=obs.vis.shape
    )

    def forward_and_vjp(response, platform):
        vis, vjp = jax.vjp(jax.jit(response), on_device(sky, platform))
        (sky_cotangent,) = jax.jit(vjp)(on_device(cotangent, platform))
        return vis, sky_cotangent

    vis_expected, grad_expected = forward_and_vjp(r_finufft, "cpu")
    vis, grad = forward_and_vjp(r_cufinufft, "cuda")

    assert vis.shape == obs.vis.shape
    assert_transform(vis, vis_expected)
    assert_transform(grad, grad_expected)
    assert np.all(np.isfinite(np.asarray(grad)))


@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
@pmp("center", ((0.0, 0.0), (1e-3, -2e-3)))
def test_backend_consistency_smallest_unit(freqs, center):
    """Single operator unit, as in MR 232's ducc/finufft test, plus cufinufft."""
    np.random.seed(42)
    npix_x, npix_y = 32, 40
    pixsize_x = np.deg2rad(1.0) / npix_x
    pixsize_y = np.deg2rad(1.5) / npix_y
    obs = generate_random_obs(
        freqs,
        20,
        [-1e2, 1e2],
        # finufft is a 2D transform, so compare on a coplanar array
        [0.0, 0.0],
        ju.polarization.PolarizationType(("I",)),
    )
    geometry = dict(
        observation=obs,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        center_x=center[0],
        center_y=center[1],
    )
    with jax.default_device(jax.devices("cpu")[0]):
        r_finufft = interferometry_response_finufft(epsilon=EPSILON, **geometry)
    with jax.default_device(jax.devices("cuda")[0]):
        r_cufinufft = CufinufftResponse(
            npix_x=npix_x, npix_y=npix_y, settings=CUFINUFFT_SETTINGS, **geometry
        )
    image = np.random.normal(size=(npix_x, npix_y))

    vis_finufft = jax.jit(r_finufft)(on_device(image, "cpu"))
    vis_cufinufft = jax.jit(r_cufinufft)(on_device(image, "cuda"))
    assert vis_cufinufft.shape == (20, len(freqs))
    assert_transform(vis_cufinufft, vis_finufft)

    if importlib.util.find_spec("jaxbind") is None:
        pytest.skip("jaxbind is required for the ducc0 comparison")
    # jaxbind registers its callback for CPU only.
    with jax.default_device(jax.devices("cpu")[0]):
        r_ducc = interferometry_response_ducc(
            npix_x=npix_x,
            npix_y=npix_y,
            do_wgridding=False,
            epsilon=EPSILON,
            nthreads=1,
            verbosity=False,
            **geometry,
        )
        vis_ducc = r_ducc(on_device(image, "cpu"))
    assert_transform(vis_cufinufft, vis_ducc)


def stokes_to_circular(iquv):
    i, q, u, v = iquv
    return np.array([i + v, q + 1j * u, q - 1j * u, i - v])


def stokes_to_linear(iquv):
    i, q, u, v = iquv
    return np.array([i + q, u + 1j * v, u - 1j * v, i - q])


@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
@pmp(
    "pol_sky,pol_channels,to_data",
    (
        (("I",), ("LL", "RR"), lambda i: np.array([i[0], i[0]])),
        (("I", "Q", "U", "V"), ("RR", "RL", "LR", "LL"), stokes_to_circular),
        (("I", "Q", "U", "V"), ("XX", "XY", "YX", "YY"), stokes_to_linear),
    ),
)
def test_point_source_at_phase_center(freqs, pol_sky, pol_channels, to_data):
    """A unit-flux point source at the phase center has constant visibilities.

    The response is linear, so any source vector works here. MR 232 draws a
    physical Stokes vector, which only matters once stokes_adder is tested.
    """
    np.random.seed(7)
    grid, obs, _, r_cufinufft = build_responses(pol_sky, pol_channels, freqs)
    source = np.random.normal(size=len(pol_sky))
    sky = np.zeros(grid.shape)
    cx, cy = grid.shape[3] // 2, grid.shape[4] // 2
    dvol = grid.spatial.dvol.to(u.rad**2).value
    sky[:, 0, :, cx, cy] = source[:, None] / dvol

    vis = np.asarray(jax.jit(r_cufinufft)(on_device(sky, "cuda")))
    vis_expected = np.broadcast_to(to_data(source)[:, None, None], obs.vis.shape)
    assert_transform(vis, vis_expected)
