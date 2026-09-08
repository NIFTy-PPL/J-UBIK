import numpy as np
import pytest

from jubik.instruments.jwst.minimization.mask_hot_pixels import (
    _hot_pixel_mask,
    _hot_star_mask_convolution,
    _hot_star_mask_from_nanpixel,
)


def test_hot_pixel_mask_none_threshold_returns_full_bool_mask():
    residual = np.array([0.1, -5.0, 3.0])
    mask = _hot_pixel_mask(residual, None)
    assert mask.dtype == bool
    assert mask.shape == residual.shape
    assert mask.all()
    # mask contract used downstream
    mask.copy()
    assert residual[mask].shape == residual.shape


def test_hot_pixel_mask_threshold():
    residual = np.array([0.1, -5.0, 3.0])
    np.testing.assert_array_equal(_hot_pixel_mask(residual, 2.0), [True, False, False])


def _field(shape=(1, 5, 5)):
    mask_og = np.ones(shape, dtype=bool)
    return mask_og


@pytest.mark.parametrize(
    "hot",
    [(0, 0), (0, 4), (4, 0), (4, 4), (0, 2), (4, 2), (2, 0), (2, 4)],
)
def test_convolution_mask_border_does_not_wrap(hot):
    mask_og = _field()
    res_var = np.zeros(mask_og.shape)
    # A single strong pixel; the cross kernel makes its 4-neighbours exceed
    # threshold, the pixel itself gets 0 from the kernel center.
    res_var[0, hot[0], hot[1]] = 10.0
    residual = res_var[mask_og]

    flat = _hot_star_mask_convolution(residual, mask_og, threshold_star_convolution=1.0)
    new_mask = mask_og.copy()
    new_mask[mask_og] = flat

    ii, jj = hot
    # Pixels within Manhattan distance 2 of the hot pixel may be masked. Anything
    # further away must stay unmasked, in particular the opposite border.
    rows, cols = np.indices(mask_og.shape[1:])
    manhattan = np.abs(rows - ii) + np.abs(cols - jj)
    assert new_mask[0][manhattan > 2].all()
    # opposite edge untouched
    opp_row = mask_og.shape[1] - 1 - ii
    opp_col = mask_og.shape[2] - 1 - jj
    if abs(opp_row - ii) > 2:
        assert new_mask[0, opp_row, :].all()
    if abs(opp_col - jj) > 2:
        assert new_mask[0, :, opp_col].all()


def test_convolution_mask_none_threshold():
    mask_og = _field()
    residual = np.ones(int(mask_og.sum()))
    flat = _hot_star_mask_convolution(residual, mask_og, None)
    assert flat.dtype == bool and flat.shape == residual.shape and flat.all()


@pytest.mark.parametrize(
    "nan_pixel",
    [(0, 0), (0, 4), (4, 0), (4, 4), (0, 2), (4, 2), (2, 0), (2, 4), (2, 2)],
)
def test_nan_neighbour_mask_border_does_not_wrap_or_raise(nan_pixel):
    mask_og = _field()
    mask_nan = np.ones(mask_og.shape, dtype=bool)
    mask_nan[0, nan_pixel[0], nan_pixel[1]] = False

    res_var = 10.0 * np.ones(mask_og.shape)
    residual = res_var[mask_og]

    flat = _hot_star_mask_from_nanpixel(residual, mask_og, mask_nan, threshold_star_nan=1.0)
    new_mask = mask_og.copy()
    new_mask[mask_og] = flat

    ii, jj = nan_pixel
    rows, cols = np.indices(mask_og.shape[1:])
    manhattan = np.abs(rows - ii) + np.abs(cols - jj)
    # exactly the in-field 4-neighbours are masked
    expected = ~(manhattan == 1)
    np.testing.assert_array_equal(new_mask[0], expected)


def test_nan_neighbour_mask_none_threshold():
    mask_og = _field()
    mask_nan = np.ones(mask_og.shape, dtype=bool)
    residual = np.ones(int(mask_og.sum()))
    flat = _hot_star_mask_from_nanpixel(residual, mask_og, mask_nan, None)
    assert flat.dtype == bool and flat.shape == residual.shape and flat.all()
