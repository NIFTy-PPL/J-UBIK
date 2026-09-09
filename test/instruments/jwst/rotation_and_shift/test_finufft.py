import numpy as np
import pytest

pytest.importorskip("jax_finufft")

from jubik.instruments.jwst.rotation_and_shift.linear_rotation_and_shift import (
    build_linear_rotation_and_shift,
)
from jubik.instruments.jwst.rotation_and_shift.nufft_rotation_and_shift import (
    build_nufft_rotation_and_shift,
)


def _box_field(maxx, maxy):
    field = np.zeros((maxx, maxy))
    field[40:60, 40:60] = 1.0
    return field


def _gaussian_field(maxx, maxy, sigma=16.0):
    i, j = np.mgrid[:maxx, :maxy]
    return np.exp(-(((i - maxx / 2) ** 2 + (j - maxy / 2) ** 2) / (2 * sigma**2)))


def _grid(maxx, maxy, indexing):
    return np.array(
        np.meshgrid(np.arange(maxx), np.arange(maxy), indexing=indexing), dtype=float
    )


def test_identity_square_ij():
    field = _box_field(128, 128)
    xy = _grid(128, 128, "ij")
    rs = build_nufft_rotation_and_shift(field.shape, field.shape, indexing="ij")
    assert np.allclose(rs(field, xy), field, atol=1e-5)


def test_identity_nonsquare_ij():
    field = _box_field(128, 160)
    xy = _grid(128, 160, "ij")
    rs = build_nufft_rotation_and_shift(field.shape, field.shape, indexing="ij")
    assert np.allclose(rs(field, xy), field, atol=1e-5)


@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_agreement_with_linear_on_smooth_field(indexing):
    # Cross-validates the two independent interpolation implementations on a
    # sub-pixel shift. The tolerance is set by the bilinear interpolation
    # error (~1/sigma^2); the nufft is spectrally accurate.
    field = _gaussian_field(128, 128)
    xy = _grid(128, 128, indexing) + 3.3

    lin = build_linear_rotation_and_shift(indexing=indexing, order=1)(field, xy)
    nft = build_nufft_rotation_and_shift(field.shape, field.shape, indexing)(field, xy)

    interior = np.s_[8:-8, 8:-8]
    assert np.allclose(lin[interior], nft[interior], atol=2e-3)


def test_constant_mode_zeroes_out_of_range():
    # The mask in mode="constant" is strict (> 2pi), so shift by a non-integer
    # to keep coordinates off the exact grid boundary, where they would wrap.
    field = _gaussian_field(128, 128)
    xy = _grid(128, 128, "ij")
    xy[0] += 100.5

    rs = build_nufft_rotation_and_shift(
        field.shape, field.shape, indexing="ij", mode="constant"
    )
    out = np.array(rs(field, xy))

    out_of_range = xy[0] > 128
    assert np.all(out[out_of_range] == 0.0)


def test_invalid_mode_raises():
    field = _gaussian_field(128, 128)
    xy = _grid(128, 128, "ij")
    rs = build_nufft_rotation_and_shift(
        field.shape, field.shape, indexing="ij", mode="mirror"
    )
    with pytest.raises(ValueError, match="wrap.*constant"):
        rs(field, xy)


@pytest.mark.xfail(
    strict=True,
    reason="nufft with indexing='xy' returns the transposed shape on non-square "
    "grids, inconsistent with the linear implementation "
    "(nufft_rotation_and_shift.py:76 swaps out_shape).",
)
def test_nonsquare_xy_shape_matches_linear():
    field = _box_field(128, 160)
    xy = _grid(128, 160, "xy")

    lin = build_linear_rotation_and_shift(indexing="xy", order=1)(field, xy)
    nft = build_nufft_rotation_and_shift(field.shape, field.shape, "xy")(field, xy)

    assert nft.shape == lin.shape
