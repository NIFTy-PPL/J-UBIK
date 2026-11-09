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
from jubik.instruments.resolve.conventions import RESPONSE_SPATIAL_UNIT
from jubik.instruments.resolve.response import (
    convert_polarization,
    dtype_float2complex,
    interferometry_response_ducc,
    interferometry_response_finufft,
)

from generate_test_obs import generate_random_obs
from jubik.instruments.resolve.data.auxiliary_table import AuxiliaryTable
from jubik.instruments.resolve.data.data_modify.frequency import (
    restrict_to_discontinuous_frequencies,
)

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


def setup_grid_obs_response(pol_sky, pol_channels, freqs):
    pol_type_sky = ju.polarization.PolarizationType(pol_sky)
    pol_type_data = ju.polarization.PolarizationType(pol_channels)
    obs = generate_random_obs(
        freqs,
        50,
        [-1e2, 1e2],
        [-5, 5],
        pol_type_data,
        fraction_flagged=0.3,
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
    pixsize_x = fov[0].to_value(RESPONSE_SPATIAL_UNIT) / npix_x
    pixsize_y = fov[1].to_value(RESPONSE_SPATIAL_UNIT) / npix_y
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
    y = jnp.asarray(
        rng.normal(size=vis_ducc.shape) + 1j * rng.normal(size=vis_ducc.shape)
    )
    gradients = []
    for response in (response_ducc, response_finufft):
        normalized = lambda x: response(x) / vol
        values, pullback = jax.vjp(normalized, image)
        adjoint_y = pullback(jnp.conj(y))[0]
        assert_allclose(
            jnp.real(jnp.vdot(values, y)),
            jnp.vdot(image, adjoint_y),
            rtol=1e-10,
            atol=1e-9,
        )
        loss = lambda x: 0.5 * jnp.sum(jnp.abs(normalized(x) - y) ** 2)
        gradients.append(jax.jit(jax.grad(loss))(image))
    assert_allclose(gradients[0], gradients[1], rtol=1e-9, atol=1e-7)


@pmp(
    "center_kwargs",
    (
        {},
        {"center_x": None, "center_y": None},
        {"center_x": None, "center_y": 0.0},
        {"center_x": 0.0, "center_y": None},
        {"center_x": None, "center_y": 2e-4},
        {"center_x": 1e-4, "center_y": None},
    ),
    ids=(
        "defaults",
        "both-none",
        "x-none-y-zero",
        "x-zero-y-none",
        "x-none-y-shift",
        "x-shift-y-none",
    ),
)
def test_finufft_missing_phase_center_skips_shift(center_kwargs):
    rng = np.random.default_rng(42)
    obs = generate_random_obs(
        np.array([1e9, 1.3e9]),
        20,
        [-1e2, 1e2],
        [0.0, 0.0],
        ju.polarization.PolarizationType(("I",)),
        rng=rng,
    )
    kwargs = dict(observation=obs, pixsize_x=1e-4, pixsize_y=2e-4, epsilon=1e-12)
    default_response = interferometry_response_finufft(**kwargs, **center_kwargs)
    zero_response = interferometry_response_finufft(
        **kwargs, center_x=0.0, center_y=0.0
    )
    image = jnp.asarray(rng.normal(size=(32, 40)))

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

    sky = jnp.asarray(np.random.default_rng(42).normal(size=grid.shape))
    res_ducc = jax.jit(r_ducc)(sky)
    res_finufft = jax.jit(r_finufft)(sky)

    assert_allclose(res_ducc, res_finufft, rtol=1e-9, atol=1e-9)


@pmp("freqs", (np.array([1e9]), np.array([1e9, 1.3e9, 2e9])))
def test_response_StokesI(freqs):
    pol_sky = ("I",)
    pol_channels = ("LL", "RR")
    grid, obs, r_ducc, r_finufft = setup_grid_obs_response(pol_sky, pol_channels, freqs)

    sky = jnp.zeros(grid.shape)
    cx = grid.shape[3] // 2
    cy = grid.shape[4] // 2
    dvol = grid.spatial.dvol.to(RESPONSE_SPATIAL_UNIT**2).value
    sky = sky.at[:, :, :, cx, cy].set(1 / dvol)
    vis_ducc = r_ducc(sky)
    vis_finufft = r_finufft(sky)
    vis_expected = np.ones(obs.vis.shape, dtype=jnp.complex128)
    assert_allclose(vis_ducc, vis_expected)
    assert_allclose(vis_finufft, vis_expected)


def make_binned_grid(spectral):
    return ju.Grid(
        spatial=ju.wcs.WcsAstropy(
            center=SkyCoord(0 * u.deg, 0 * u.deg),
            shape=(32, 40),
            fov=u.Quantity([1.0, 1.5], u.deg),
        ),
        spectral=spectral,
    )


@pytest.fixture(params=("ducc", "finufft"))
def backend_settings(request):
    if request.param == "ducc":
        return rve.parse.Ducc0Settings(
            epsilon=1e-10, do_wgridding=False, nthreads=1, verbosity=False
        )
    return rve.parse.FinufftSettings(epsilon=1e-10)


def make_frequency_obs(freqs, fraction_flagged=0.0):
    return generate_random_obs(
        np.asarray(freqs) * 1e9,
        20,
        [-100, 100],
        [0, 0],
        ju.polarization.PolarizationType(("I",)),
        fraction_flagged=fraction_flagged,
        rng=np.random.default_rng(42),
    )


@pmp("position_angle", (0.0, 90.0, 30.0), ids=("north", "east", "northeast"))
def test_response_phase_center_from_sky_coordinates(backend_settings, position_angle):
    sky_center = SkyCoord(1.0 * u.rad, 0.5 * u.rad, frame="icrs")
    offset = 3e-4
    phase_center = sky_center.directional_offset_by(
        position_angle * u.deg, offset * u.rad
    )
    # Synthetic observation with w=0 (coplanar array)
    obs = make_frequency_obs([1.0, 1.2])
    field = AuxiliaryTable(
        {"REFERENCE_DIR": np.array([[[phase_center.ra.rad, phase_center.dec.rad]]])}
    )
    obs = rve.Observation(
        obs.antenna_positions,
        obs.vis_val,
        obs.weight_val,
        obs.legacy_polarization,
        obs.freq,
        {"FIELD": field},
    )
    grid = ju.Grid(
        spatial=ju.wcs.WcsAstropy(
            center=sky_center, shape=(32, 40), fov=u.Quantity([1.0, 1.5], u.deg)
        ),
        spectral=ju.color.Color(np.array([0.8, 1.5]) * u.GHz),
    )
    cy, cx = (size // 2 for size in grid.shape[-2:])
    vol = grid.spatial.dvol.to_value(RESPONSE_SPATIAL_UNIT**2)
    sky = jnp.zeros(grid.shape).at[0, 0, 0, cy, cx].set(1 / vol)

    # Known angular offsets independently pin the coordinate helper's axis order
    # and the phase sign. Coplanar baselines isolate the phase-center shift.
    angle = (position_angle * u.deg).to_value(u.rad)
    center_x, center_y = offset * np.sin(angle), offset * np.cos(angle)
    uvw = obs.uvw[:, None, :] * obs.freq[None, :, None] / c.value
    expected = np.exp(-2j * np.pi * (uvw[..., 0] * center_x + uvw[..., 1] * center_y))
    response = rve.interferometry_response(obs, grid, backend_settings)
    for result in (response(sky), jax.jit(response)(sky)):
        assert_allclose(result, expected[None, ...], rtol=1e-9, atol=1e-9)


@pmp("circular", (True, False), ids=("circular", "linear"))
def test_public_response_adjoint_and_gradient(circular):
    pol = ju.polarization.PolarizationType
    grid = ju.Grid(
        spatial=ju.wcs.WcsAstropy(
            center=SkyCoord(0 * u.deg, 0 * u.deg),
            shape=(32, 40),
            fov=u.Quantity([1.0, 1.5], u.deg),
        ),
        spectral=ju.color.Color(np.array([0.8, 1.5, 1.8, 2.5]) * u.GHz),
        polarization=pol.IQUV,
    )
    rng = np.random.default_rng(42)
    obs = generate_random_obs(
        np.array([1.0, 1.2, 2.0, 2.2]) * 1e9,
        20,
        [-100, 100],
        [0, 0],
        pol.RR_RL_LR_LL if circular else pol.XX_XY_YX_YY,
        rng=rng,
    )
    sky = jnp.asarray(rng.normal(size=grid.shape))
    y = jnp.asarray(rng.normal(size=obs.vis.shape) + 1j * rng.normal(size=obs.vis.shape))
    vol = grid.spatial.dvol.to_value(RESPONSE_SPATIAL_UNIT**2)
    gradients = []
    for settings in (
        rve.parse.Ducc0Settings(
            epsilon=1e-12, do_wgridding=False, nthreads=1, verbosity=False
        ),
        rve.parse.FinufftSettings(epsilon=1e-12),
    ):
        response = rve.interferometry_response(obs, grid, settings)
        normalized = lambda x: response(x) / vol
        values, pullback = jax.vjp(normalized, sky)
        # Real Stokes skies require the real part of the complex pairing.
        adjoint_y = pullback(jnp.conj(y))[0]
        assert_allclose(
            jnp.real(jnp.vdot(values, y)),
            jnp.vdot(sky, adjoint_y),
            rtol=1e-10,
            atol=1e-8,
        )
        loss = lambda x: 0.5 * jnp.sum(jnp.abs(normalized(x) - y) ** 2)
        gradient = jax.jit(jax.grad(loss))(sky)
        gradients.append(gradient)
        # No observed channel uses the middle sky bin, so its gradient is zero.
        assert_allclose(gradient[:, :, 1], 0, rtol=0, atol=0)
        # A central finite difference independently checks the full loss gradient.
        direction = jnp.asarray(rng.normal(size=grid.shape))
        direction = direction / jnp.linalg.norm(direction)
        step = 1e-4
        finite_difference = (loss(sky + step * direction) - loss(sky - step * direction)) / (
            2 * step
        )
        assert_allclose(
            finite_difference, jnp.vdot(gradient, direction), rtol=1e-6, atol=1e-5
        )
    assert_allclose(gradients[0], gradients[1], rtol=1e-9, atol=1e-7)


def test_response_frequency_bins_and_empty_bin(backend_settings):
    grid = make_binned_grid(ju.color.Color(np.array([0.8, 1.5, 1.8, 2.5]) * u.GHz))
    obs = make_frequency_obs([1.0, 1.2, 2.0, 2.2])
    # The response implementation assigns the observed frequencies into the defined bins
    # of the sky frequency grid. Here The first two are assigned to the first sky bin,
    # the last two to the last sky bin.
    response = rve.interferometry_response(obs, grid, backend_settings)
    vol = grid.spatial.dvol.to_value(RESPONSE_SPATIAL_UNIT**2)
    # Spatial axes are canonical (Dec, RA); derive the center from the sky shape.
    cy, cx = (size // 2 for size in grid.shape[-2:])
    sky = (
        jnp.zeros(grid.shape).at[0, 0, :, cy, cx].set(jnp.array([2.0, 99.0, 5.0]) / vol)
    )
    result = jax.jit(response)(sky)
    # Because of the bin assignment the we expect that the flux for the first two
    # observed frequencies comes from the first sky bin and the last to from the last
    # sky bin.
    expected = np.broadcast_to([2.0, 2.0, 5.0, 5.0], obs.vis.shape)
    assert_allclose(result, expected, rtol=1e-9, atol=1e-9)
    # Changing the empty bin must have no effect on any channel.
    # Checks if reassigment of middle sky bin does not change the prediction of the
    # other two sky bins.
    assert_allclose(response(sky.at[:, :, 1].set(0)), result, rtol=1e-12, atol=1e-12)


def test_response_discontinuous_frequency_restriction(backend_settings):
    spectral = ju.color.Color(np.array([[0.8, 1.5], [1.8, 2.5]]) * u.GHz)
    grid = make_binned_grid(spectral)
    obs = make_frequency_obs([1.0, 1.2, 1.6, 2.0, 2.2])
    # Frequency 1.6 cannot be assigned to any sky bin as the spectral sky grid
    # is discontinous. Thus only the observed frequencies with indices 0, 1, 3 and 4
    # survive.
    with pytest.raises(RuntimeError, match="restrict_to_discontinuous_frequencies"):
        rve.interferometry_response(obs, grid, backend_settings)
    restricted = restrict_to_discontinuous_frequencies(obs, spectral)
    indices = [0, 1, 3, 4]
    # Tests if the observed frequency is removed from the frequency, visibility
    #  and data array by data modification routine
    assert_allclose(restricted.freq, obs.freq[indices])
    assert_allclose(restricted.vis, obs.vis[..., indices])
    assert_allclose(restricted.weight, obs.weight[..., indices])
    response = rve.interferometry_response(restricted, grid, backend_settings)
    vol = grid.spatial.dvol.to_value(RESPONSE_SPATIAL_UNIT**2)
    # Keep the source centered so this checks frequency assignment without phase.
    cy, cx = (size // 2 for size in grid.shape[-2:])
    sky = jnp.zeros(grid.shape).at[0, 0, :, cy, cx].set(jnp.array([2.0, 5.0]) / vol)
    # Checks correct prediction assignment as in test_response_frequency_bins_and_empty_bin
    assert_allclose(
        jax.jit(response)(sky),
        np.broadcast_to([2.0, 2.0, 5.0, 5.0], restricted.vis.shape),
        rtol=1e-9,
        atol=1e-9,
    )


@pmp(
    "dtype, expected_dtype",
    ((jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)),
)
def test_response_dtype_and_flags(backend_settings, dtype, expected_dtype):
    # DUCC's single-precision kernels require epsilon > 1e-5.
    if dtype == jnp.float32:
        if isinstance(backend_settings, rve.parse.Ducc0Settings):
            backend_settings = rve.parse.Ducc0Settings(
                epsilon=2e-5, do_wgridding=False, nthreads=1, verbosity=False
            )
    grid = make_binned_grid(ju.color.Color(np.array([0.8, 1.5]) * u.GHz))
    obs = make_frequency_obs([1.0, 1.2], fraction_flagged=0.5)
    # Building unflagged as reference ground truth as response does not remove
    # flagged visibilities by itself
    unflagged = make_frequency_obs([1.0, 1.2])
    mask = obs.weight == 0
    assert np.any(mask) and np.any(~mask)
    vol = grid.spatial.dvol.to_value(RESPONSE_SPATIAL_UNIT**2)
    # Point source sky with unit flux
    sky = (
        jnp.zeros(grid.shape, dtype=dtype)
        .at[0, 0, 0, 17, 18]
        .set(jnp.asarray(1 / vol, dtype=dtype))
    )
    result = jax.jit(rve.interferometry_response(obs, grid, backend_settings))(sky)
    reference = rve.interferometry_response(unflagged, grid, backend_settings)(sky)
    assert result.dtype == expected_dtype
    # The response predicts all samples; weights belong to the likelihood.
    assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
    # Checks if unit flux is roughly preserved for the visibility prediction
    assert_allclose(
        np.abs(np.asarray(result)[mask]),
        1.0,
        rtol=1e-4,
        atol=1e-6,
    )


@pmp("dtype, expected", ((np.float32, np.complex64), (np.float64, np.complex128)))
def test_dtype_float2complex(dtype, expected):
    assert dtype_float2complex(dtype) == expected


@pmp("dtype", (np.int32, np.int64, np.bool_, np.complex64))
def test_dtype_float2complex_rejects_nonfloat(dtype):
    with pytest.raises(ValueError):
        dtype_float2complex(dtype)


@pmp("inp_pol", tuple(ju.polarization.PolarizationType), ids=lambda pol: pol.name)
@pmp("out_pol", tuple(ju.polarization.PolarizationType), ids=lambda pol: pol.name)
def test_convert_polarization_all_types(inp_pol, out_pol):
    """Check supported conversions and explicit rejection for all enum pairs."""
    pol = ju.polarization.PolarizationType
    values = np.arange(1, len(inp_pol) * 6 + 1).reshape(len(inp_pol), 2, 3)
    # Stokes visibilities can be complex; vary both parts across all axes.
    inp = jnp.asarray(values + 1j * values[..., ::-1])

    # Unlisted pairs are currently unsupported, even if mathematically possible.
    # Correlation inputs have no implemented conversion branch: single channels
    # cannot generally recover Stokes, while complete sets need an inverse that
    # is not implemented. Even correlation identity conversions are rejected.
    expected = None
    if inp_pol == pol.I:
        if out_pol in (pol.I, pol.RR, pol.LL, pol.XX, pol.YY):
            # With Q=U=V=0, each parallel-hand channel equals I; I->I is identity.
            expected = np.asarray(inp)
        elif out_pol in (pol.LL_RR, pol.RR_LL, pol.XX_YY, pol.YY_XX):
            # Both parallel hands equal I, so the polarization axis is duplicated.
            # Reversing their order has no effect for an I-only input.
            expected = np.repeat(np.asarray(inp), 2, axis=0)
        # Expansion of I into full Stokes or full correlation sets is not
        # implemented, despite being possible with Q=U=V=0.
    elif inp_pol == pol.IQUV:
        i, q, u, v = np.asarray(inp)
        if out_pol == pol.RR_RL_LR_LL:
            # Circular parallel hands mix I,V; cross hands mix Q,U.
            expected = np.stack((i + v, q + 1j * u, q - 1j * u, i - v))
        elif out_pol == pol.XX_XY_YX_YY:
            # Linear parallel hands mix I,Q; cross hands mix U,V.
            expected = np.stack((i + q, u + 1j * v, u - 1j * v, i - q))
        # Only complete correlation outputs are implemented for IQUV input;
        # subsets, I extraction, and IQUV identity currently raise as well.

    convert = lambda x: convert_polarization(x, inp_pol, out_pol)
    if expected is None:
        # Unsupported pairs must fail explicitly, both directly and under JIT.
        with pytest.raises(NotImplementedError, match="not implemented"):
            convert(inp)
        with pytest.raises(NotImplementedError, match="not implemented"):
            jax.jit(convert)(inp)
    else:
        # Supported pairs must preserve the non-polarization axes and agree
        # with the explicit formulas, both directly and under JIT.
        for result in (convert(inp), jax.jit(convert)(inp)):
            assert result.shape == expected.shape
            assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_ducc_wgridding_off_center_point_source():
    obs = generate_random_obs(
        np.array([1e9, 1.5e9]),
        20,
        [-100, 100],
        [-1000, 1000],
        ju.polarization.PolarizationType(("I",)),
        rng=np.random.default_rng(42),
    )
    px, py = 5e-4, 7e-4
    l, m = 5 * px, -3 * py
    n = np.sqrt(1 - l * l - m * m)
    response = interferometry_response_ducc(
        obs,
        32,
        40,
        px,
        py,
        do_wgridding=True,
        epsilon=1e-10,
        nthreads=1,
        verbosity=False,
    )
    image = jnp.zeros((32, 40)).at[21, 17].set(1 / (px * py))
    uvw = obs.uvw[:, None, :] * obs.freq[None, :, None] / c.value
    # DUCC's measurement-set convention uses -w(n-1) inside the negative phase.
    expected = (
        np.exp(
            -2j * np.pi * (uvw[..., 0] * l - uvw[..., 1] * m - uvw[..., 2] * (n - 1))
        )
        / n
    )
    assert_allclose(jax.jit(response)(image), expected, rtol=1e-9, atol=1e-9)


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
    dvol = grid.spatial.dvol.to(RESPONSE_SPATIAL_UNIT**2).value
    sky = sky.at[:, 0, :, cx, cy].set(source[np.newaxis].T / dvol)
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


def test_response_rejects_beam_of_wrong_shape():
    freqs = np.array([1e9])
    obs = generate_random_obs(
        freqs, 10, [-1e2, 1e2], [-5, 5], ju.polarization.PolarizationType(("LL", "RR"))
    )
    spatial = ju.wcs.WcsAstropy(
        center=SkyCoord(0 * u.deg, 0 * u.deg), shape=(8, 8), fov=u.Quantity([1, 1], u.deg)
    )
    grid = ju.Grid(spatial=spatial, spectral=ju.color.Color.from_central_frequencies(freqs))
    with pytest.raises(ValueError, match="shape of the grid"):
        rve.interferometry_response(
            obs, grid, rve.parse.FinufftSettings(epsilon=1e-9), beam=np.ones((8, 8))
        )
