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
