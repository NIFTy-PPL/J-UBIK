"""A point source through the star and target paths (PSF + integration) must land
at its WCS position, for odd and even centered PSF kernels (issue #130)."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.instruments.jwst.alignment.star_alignment import Star
from jubik.instruments.jwst.alignment.star_model import (
    bilinear_point_source_evaluation,
)
from jubik.instruments.jwst.integration.integration import integration_factory
from jubik.instruments.jwst.psf.psf_operator import PsfStatic, _center_even_axes
from jubik.wcs.wcs_astropy import WcsAstropy
from jubik.wcs.wcs_subsample_centers import subsample_pixel_centers

import nifty.re as jft

DATA_WCS = WcsAstropy(SkyCoord(10 * u.deg, -5 * u.deg), shape=40, fov=40 * u.arcsec)
SOURCE_XY = (17.3, 21.8)  # detector (x, y)
SOURCE = DATA_WCS.pixel_to_world(*SOURCE_XY)
TARGET_BOUNDS = (5, 35, 3, 33)  # min_row, max_row, min_column, max_column
SIGMA = 1.5  # native pixels, PSF and target source width


def _kernel(size, subsample):
    """Gaussian centered on the array center (size - 1) / 2, as STPSF returns it."""
    size = np.broadcast_to(size, 2)
    ry, rx = (np.arange(n) - (n - 1) / 2 for n in size)
    k = np.exp(-0.5 * (ry[:, None] ** 2 + rx[None, :] ** 2) / (SIGMA * subsample) ** 2)
    return k / k.sum()


def _response(fine, kernel, subsample):
    psf = PsfStatic(jft.ShapeWithDtype(fine.shape, float), kernel)
    integrate = integration_factory(u.Unit("Jy"), fine.shape, subsample)
    return np.asarray(integrate(psf((fine, {}))))


def _centroid_xy(image, min_row, min_column):
    y, x = np.indices(image.shape)
    return np.array(
        [(image * x).sum() + min_column * image.sum(), (image * y).sum() + min_row * image.sum()]
    ) / image.sum()


def _target(kernel, s):
    centers = subsample_pixel_centers(TARGET_BOUNDS, DATA_WCS, s)
    r = centers.separation(SOURCE).to_value(u.arcsec)  # 1 arcsec data pixels
    fine = np.exp(-0.5 * r**2 / SIGMA**2)
    return _centroid_xy(_response(fine, kernel, s), TARGET_BOUNDS[0], TARGET_BOUNDS[2])


def _star(kernel, s):
    min_row, _, min_column, _ = Star._get_bounding_indices(SOURCE_XY, (15, 15), (40, 40))
    location = Star.skycoord_to_subpixel(SOURCE, DATA_WCS, min_row, min_column, s)
    fine = bilinear_point_source_evaluation(np.zeros((15 * s, 15 * s)), np.array(location), 1.0)
    return _centroid_xy(_response(np.asarray(fine), kernel, s), min_row, min_column)


@pytest.mark.parametrize("s", [1, 2, 3])
@pytest.mark.parametrize("parity", [0, 1])
def test_star_and_target_land_on_the_wcs_position(s, parity):
    kernel = _kernel(10 * s + parity, s)
    np.testing.assert_allclose(_star(kernel, s), SOURCE_XY, atol=2e-3)
    np.testing.assert_allclose(_target(kernel, s), SOURCE_XY, atol=2e-3)


def _centroid_yx(k):
    y, x = np.indices(k.shape)
    return np.array([(k * y).sum(), (k * x).sum()]) / k.sum()


@pytest.mark.parametrize("shape", [(11, 11), (10, 10), (11, 10), (10, 11)])
def test_kernel_recentered_per_axis(shape):
    kernel = _kernel(shape, 2)
    centered = _center_even_axes(kernel)
    assert centered.shape == tuple(n + 1 - n % 2 for n in shape)
    np.testing.assert_allclose(_centroid_yx(centered), np.array(centered.shape) // 2, atol=1e-9)
    np.testing.assert_allclose(centered.sum(), kernel.sum(), rtol=1e-12)
    if shape == (11, 11):
        assert centered is kernel  # odd kernels are left untouched


def test_stacked_kernels_recentered():
    kernels = np.stack([_kernel(10, 2), 2 * _kernel(10, 2)])
    centered = _center_even_axes(kernels)
    np.testing.assert_allclose(centered[1], 2 * _center_even_axes(kernels[0]))
    np.testing.assert_allclose(centered.sum(axis=(1, 2)), [1.0, 2.0])


def test_recentering_ringing_on_truncated_airy_kernel():
    from scipy.special import j1

    # Airy pattern with its first dark ring 4 fine pixels out, cut at its edge.
    r = np.arange(16) - 7.5
    x = 3.8317 * np.hypot(r[:, None], r[None, :]) / 4
    kernel = (2 * j1(x) / x) ** 2
    centered = _center_even_axes(kernel)
    assert centered.min() / centered.max() > -1e-3
