import numpy as np
import pytest

pytest.importorskip("jax_finufft")

from jubik.instruments.jwst.rotation_and_shift.linear_rotation_and_shift import (
    build_linear_rotation_and_shift,
)
from jubik.instruments.jwst.rotation_and_shift.nufft_rotation_and_shift import (
    build_nufft_rotation_and_shift,
)


def _box_field(maxy, maxx):
    field = np.zeros((maxy, maxx))
    field[40:60, 40:60] = 1.0
    return field


def _gaussian_field(maxy, maxx, sigma=16.0):
    yy, xx = np.mgrid[:maxy, :maxx]
    return np.exp(
        -(((yy - maxy / 2) ** 2 + (xx - maxx / 2) ** 2) / (2 * sigma**2))
    )


def _grid_yx(maxy, maxx):
    return np.array(
        np.meshgrid(np.arange(maxy), np.arange(maxx), indexing="ij"),
        dtype=float,
    )


def test_identity_square():
    field = _box_field(128, 128)
    centers_yx = _grid_yx(*field.shape)
    rs = build_nufft_rotation_and_shift(field.shape, field.shape)
    assert np.allclose(rs(field, centers_yx), field, atol=1e-5)


def test_identity_nonsquare():
    field = _box_field(128, 160)
    centers_yx = _grid_yx(*field.shape)
    rs = build_nufft_rotation_and_shift(field.shape, field.shape)
    assert np.allclose(rs(field, centers_yx), field, atol=1e-5)


def test_agreement_with_linear_on_smooth_field():
    # Cross-validates the two independent interpolation implementations on a
    # sub-pixel shift. The tolerance is set by the bilinear interpolation
    # error (~1/sigma^2); the NUFFT is spectrally accurate.
    field = _gaussian_field(128, 160)
    centers_yx = _grid_yx(*field.shape) + 3.3

    lin = build_linear_rotation_and_shift(
        out_shape=field.shape, order=1
    )(field, centers_yx)
    nft = build_nufft_rotation_and_shift(
        field.shape, field.shape
    )(field, centers_yx)

    interior = np.s_[8:-8, 8:-8]
    assert np.allclose(lin[interior], nft[interior], atol=2e-3)


def test_constant_mode_zeroes_out_of_range():
    # The mask in mode="constant" is strict (> 2pi), so shift by a non-integer
    # to keep coordinates off the exact grid boundary, where they would wrap.
    field = _gaussian_field(128, 160)
    centers_yx = _grid_yx(*field.shape)
    centers_yx[0] += 100.5

    rs = build_nufft_rotation_and_shift(
        field.shape, field.shape, mode="constant"
    )
    out = np.array(rs(field, centers_yx))

    out_of_range = centers_yx[0] > field.shape[0]
    assert np.all(out[out_of_range] == 0.0)


def test_invalid_mode_raises():
    field = _gaussian_field(128, 160)
    centers_yx = _grid_yx(*field.shape)
    rs = build_nufft_rotation_and_shift(
        field.shape, field.shape, mode="mirror"
    )
    with pytest.raises(ValueError, match="wrap.*constant"):
        rs(field, centers_yx)


@pytest.mark.parametrize(
    "builder",
    [
        lambda shape: build_linear_rotation_and_shift(out_shape=shape),
        lambda shape: build_nufft_rotation_and_shift(shape, shape),
    ],
    ids=["linear", "nufft"],
)
def test_mismatched_coordinate_shape_raises(builder):
    field = _gaussian_field(32, 48)
    transposed_centers_yx = _grid_yx(48, 32)

    with pytest.raises(
        ValueError,
        match=r"trailing shape \(48, 32\) does not match out_shape \(32, 48\)",
    ):
        builder(field.shape)(field, transposed_centers_yx)


def _fourier_reference(field, centers_yx):
    # Direct trigonometric-interpolation sum, independent of FINUFFT.
    ky, kx = (np.fft.fftfreq(n) for n in field.shape)
    coeffs = np.fft.ifft2(field)
    phase_y = np.exp(-2j * np.pi * np.multiply.outer(centers_yx[0], ky))
    phase_x = np.exp(-2j * np.pi * np.multiply.outer(centers_yx[1], kx))
    return np.einsum("...a,...b,ab->...", phase_y, phase_x, coeffs).real


SHAPES = [(8, 8), (9, 9), (8, 9), (9, 8), (16, 21)]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("kind", ["constant", "random"])
def test_identity_parity(shape, kind):
    rng = np.random.default_rng(42)
    field = np.ones(shape) if kind == "constant" else rng.normal(size=shape)
    rs = build_nufft_rotation_and_shift(shape, shape)
    out = np.asarray(rs(field, _grid_yx(*shape)))
    np.testing.assert_allclose(out, field, atol=1e-4)


@pytest.mark.parametrize("shape", [(9, 9), (9, 12), (12, 9)], ids=str)
def test_noninteger_shift_matches_fourier_reference(shape):
    field = np.random.default_rng(0).normal(size=shape)
    centers_yx = _grid_yx(*shape) + np.array([0.37, -1.6])[:, None, None]
    rs = build_nufft_rotation_and_shift(shape, shape, mode="wrap")
    np.testing.assert_allclose(
        rs(field, centers_yx), _fourier_reference(field, centers_yx), atol=1e-4
    )


def test_nonsquare_out_shape_matches_linear_and_reference():
    # sky_shape != out_shape, both non-square: output is out_shape in yx
    # order (like the linear path) and values follow the yx coordinates.
    sky_shape, out_shape = (12, 9), (5, 7)
    field = np.random.default_rng(1).normal(size=sky_shape)
    centers_yx = 0.5 + np.stack(
        [
            np.broadcast_to(1.3 * np.arange(5)[:, None], out_shape),
            np.broadcast_to(0.9 * np.arange(7)[None, :], out_shape),
        ]
    )
    nft = np.asarray(
        build_nufft_rotation_and_shift(sky_shape, out_shape)(field, centers_yx)
    )
    lin = build_linear_rotation_and_shift(out_shape)(field, centers_yx)
    assert nft.shape == lin.shape == out_shape
    np.testing.assert_allclose(
        nft, _fourier_reference(field, centers_yx), atol=1e-4
    )


def _stacked_centers(out_shape, n_exposures=3):
    # Rotated, shifted exposures; the last one runs past the sky edge so
    # mode="constant" masks part of it.
    rng = np.random.default_rng(3)
    base = _grid_yx(*out_shape) * 0.7 + 1.1
    centers = []
    for i in range(n_exposures):
        theta = rng.uniform(-0.3, 0.3)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        centers.append(np.einsum("ij,j...->i...", rot, base) + 4.0 * i)
    return np.stack(centers)


def test_stacked_exposures_one_call_matches_per_exposure():
    import jax

    from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
        NufftConfig,
    )
    from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
        Coordinates,
    )
    from jubik.instruments.jwst.rotation_and_shift.rotation_and_shift import (
        build_rotation_and_shift,
    )

    sky_shape, out_shape = (24, 20), (13, 17)
    field = np.random.default_rng(4).normal(size=sky_shape)
    centers = _stacked_centers(out_shape)
    assert np.any(centers > np.array(sky_shape)[:, None, None])  # masking exercised

    rs = build_nufft_rotation_and_shift(sky_shape, out_shape)
    per_exposure = lambda f, c: jax.numpy.stack([rs(f, ci) for ci in c])

    one_call = build_rotation_and_shift(
        {"sky": jax.ShapeDtypeStruct(sky_shape, field.dtype)},
        Coordinates(centers),
        NufftConfig(),
    )
    np.testing.assert_allclose(
        one_call({"sky": field}), per_exposure(field, centers), rtol=0, atol=1e-12
    )

    cotangent = np.random.default_rng(5).normal(size=centers.shape[:1] + out_shape)
    _, vjp_one = jax.vjp(rs, field, centers)
    _, vjp_per = jax.vjp(per_exposure, field, centers)
    for g_one, g_per in zip(vjp_one(cotangent), vjp_per(cotangent)):
        np.testing.assert_allclose(g_one, g_per, rtol=0, atol=1e-12)


def _inside(centers, sky_shape):
    upper = np.array(sky_shape)[:, None, None]
    return np.all((centers >= 0) & (centers <= upper), axis=-3)


@pytest.mark.parametrize("slack", [0, 50, -50])  # tight, room to spare, overflow
def test_capacity_gives_every_points_result(slack):
    import jax

    sky_shape, out_shape = (24, 20), (13, 17)
    field = np.random.default_rng(4).normal(size=sky_shape)
    centers = _stacked_centers(out_shape)
    inside = _inside(centers, sky_shape)
    # some points outside; the first inside, as the unused slots repeat it
    assert 0 < inside.sum() < inside.size and inside.flat[0]
    capacity = int(inside.sum()) + slack

    every = build_nufft_rotation_and_shift(sky_shape, out_shape)
    some = build_nufft_rotation_and_shift(sky_shape, out_shape, capacity=capacity)
    np.testing.assert_allclose(some(field, centers), every(field, centers), atol=1e-12)

    cotangent = np.random.default_rng(5).normal(size=centers.shape[:1] + out_shape)
    _, vjp_some = jax.vjp(some, field, centers)
    _, vjp_every = jax.vjp(every, field, centers)
    for g_some, g_every in zip(vjp_some(cotangent), vjp_every(cotangent)):
        np.testing.assert_allclose(g_some, g_every, rtol=0, atol=1e-12)
    tangents = (np.ones(sky_shape), np.ones_like(centers))
    np.testing.assert_allclose(
        jax.jvp(some, (field, centers), tangents)[1],
        jax.jvp(every, (field, centers), tangents)[1],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("slack, every_point", [(0, False), (-1, True)])
def test_capacity_interpolates_only_the_inside_points(monkeypatch, slack, every_point):
    import jax_finufft

    sky_shape, out_shape = (24, 20), (13, 17)
    field = np.random.default_rng(4).normal(size=sky_shape)
    centers = _stacked_centers(out_shape)
    inside = _inside(centers, sky_shape)
    nufft2 = jax_finufft.nufft2

    # an interpolation at every point adds 1000, so the output tells which ran
    def marked(f, y, x):
        return nufft2(f, y, x) + 1000.0 * (y.shape[0] == inside.size)

    monkeypatch.setattr(jax_finufft, "nufft2", marked)
    some = build_nufft_rotation_and_shift(
        sky_shape, out_shape, capacity=int(inside.sum()) + slack
    )
    out = np.asarray(some(field, centers))
    assert np.all(out[~inside] == 0)
    assert np.all(np.abs(out[inside]) > 500) == every_point
    assert np.all(np.abs(out[inside]) < 500) == (not every_point)


def test_builder_sizes_the_capacity_from_the_coordinates(monkeypatch):
    import jax

    import jubik.instruments.jwst.rotation_and_shift.rotation_and_shift as rs
    from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
        NufftConfig,
    )
    from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
        Coordinates,
    )

    sky_shape, out_shape = (24, 20), (13, 17)
    centers = _stacked_centers(out_shape)
    seen = {}
    real = rs.build_nufft_rotation_and_shift
    monkeypatch.setattr(
        rs,
        "build_nufft_rotation_and_shift",
        lambda **kw: seen.update(kw) or real(**kw),
    )
    rs.build_rotation_and_shift(
        {"sky": jax.ShapeDtypeStruct(sky_shape, float)},
        Coordinates(centers),
        NufftConfig(),
    )
    assert seen["capacity"] == int(np.ceil(1.05 * _inside(centers, sky_shape).sum()))


def _cosine(freq_yx, shape_or_points):
    """cos(2 pi (f_y y + f_x x)) on a grid shape or at ``(2, ...)`` points."""
    y, x = (
        np.mgrid[: shape_or_points[0], : shape_or_points[1]]
        if isinstance(shape_or_points, tuple)
        else shape_or_points
    )
    return np.cos(2 * np.pi * (freq_yx[0] * y + freq_yx[1] * x))


def test_band_limit_keeps_the_band_and_drops_the_rest():
    # Grid harmonics on a non-square sky, an anisotropic band (0.2, 0.25)
    # cycles per pixel. Kept: one cosine well inside, one at 0.22 along x (only
    # inside because the x half-axis is 0.25, not 0.2). Dropped: one at 0.30
    # along x, and one inside the band's bounding box but outside the ellipse
    # (1.17 > 1), which only the elliptical mask, not the crop, removes.
    shape, band = (96, 128), (0.2, 0.25)
    kept = (10 / 96, 6 / 128), (0.0, 28 / 128)
    dropped = (0.0, 38 / 128), (17 / 96, 20 / 128)
    field = sum(_cosine(f, shape) for f in kept + dropped)
    points = (
        np.random.default_rng(6).uniform(0, 1, (2, 9, 11))
        * np.array(shape)[:, None, None]
    )

    limited = build_nufft_rotation_and_shift(shape, (9, 11), band_limit=band)
    expected = sum(_cosine(f, points) for f in kept)
    np.testing.assert_allclose(limited(field, points), expected, rtol=0, atol=1e-5)
    full = build_nufft_rotation_and_shift(shape, (9, 11))  # control: all four
    every = sum(_cosine(f, points) for f in kept + dropped)
    np.testing.assert_allclose(full(field, points), every, rtol=0, atol=1e-5)


# a Nyquist band that the unit conversion rounds to just below 0.5 must raise too
@pytest.mark.parametrize(
    "band", [(0.5, 0.2), (np.nextafter(0.5, 0), 0.2), (0.2, 0.0), (-0.1, 0.2)]
)
def test_band_limit_must_lie_below_nyquist(band):
    with pytest.raises(ValueError, match="band_limit"):
        build_nufft_rotation_and_shift((16, 16), (4, 4), band_limit=band)


def test_band_limit_config_is_cycles_per_arcsec():
    import jax
    from astropy import units as u

    from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
        NufftConfig,
    )
    from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
        Coordinates,
    )
    from jubik.instruments.jwst.rotation_and_shift.rotation_and_shift import (
        build_rotation_and_shift,
    )

    # 0.01" x 0.02" sky pixels: 20 cycles per arcsec is (0.2, 0.4) cycles per
    # pixel in (y, x), so a cosine at 0.3 cycles per pixel passes along x and
    # one at 0.3 along y is removed (a y/x mix-up swaps the two).
    shape, scales = (100, 100), [0.01, 0.02] * u.arcsec
    passes, removed = (0.0, 30 / 100), (30 / 100, 0.0)
    field = _cosine(passes, shape) + _cosine(removed, shape)
    points = np.random.default_rng(7).uniform(0, 100, (2, 5, 6))
    config = NufftConfig.from_yaml_dict({"band_limit": 20.0})
    assert config.band_limit == 20.0
    rs = build_rotation_and_shift(
        {"sky": jax.ShapeDtypeStruct(shape, field.dtype)},
        Coordinates(points),
        config,
        sky_pixel_scales_yx=scales.to(u.deg),
    )
    np.testing.assert_allclose(
        rs({"sky": field}), _cosine(passes, points), rtol=0, atol=1e-5
    )
    with pytest.raises(ValueError, match="pixel scales"):
        build_rotation_and_shift(
            {"sky": jax.ShapeDtypeStruct(shape, field.dtype)},
            Coordinates(points),
            config,
        )
