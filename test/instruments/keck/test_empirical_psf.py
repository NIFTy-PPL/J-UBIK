"""Empirical NIRC2 PSF kernels from star cutouts (issue #164)."""

import nifty.re as jft
import numpy as np
import pytest
from astropy.io import fits
from scipy.special import erf

from jubik.instruments.jwst.psf.psf_operator import PsfStatic
from jubik.instruments.keck.empirical_psf import (
    kernel_from_fits,
    kernel_from_star_cutout,
)

BACKGROUND = 7.25
FLUX = 1000.0
OFFSET = (0.37, -0.21)  # (y, x) star offset from the cutout middle, pixels
ORIGIN = (512, 300)  # (x, y) detector pixel of cutout[0, 0]


def _integrated_gaussian(n, centre, sigma):
    """A 1-D unit Gaussian integrated over pixels [i - 0.5, i + 0.5]."""
    edges = (np.arange(n + 1) - 0.5 - centre) / (np.sqrt(2) * sigma)
    return 0.5 * np.diff(erf(edges))


def _psf(shape, centre):
    """Pixel-integrated core + halo PSF, point-symmetric about `centre` (y, x)."""
    image = np.zeros(shape)
    for weight, (sy, sx) in ((0.7, (1.4, 1.8)), (0.3, (3.0, 3.5))):
        gy = _integrated_gaussian(shape[0], centre[0], sy)
        gx = _integrated_gaussian(shape[1], centre[1], sx)
        image += weight * np.outer(gy, gx)
    return image


def _centroid(image):
    y, x = np.indices(image.shape)
    return np.array([(y * image).sum(), (x * image).sum()]) / image.sum()


def _star(shape):
    middle = (np.array(shape) - 1) / 2
    cutout = FLUX * _psf(shape, middle + OFFSET) + BACKGROUND
    truth = _psf(shape, middle)
    return cutout, truth / truth.sum(), middle


def _build(cutout, mask):
    return kernel_from_star_cutout(
        cutout, mask, exposure="n0042", origin=ORIGIN, pixel_scale=0.009942
    )


@pytest.mark.parametrize("shape", [(41, 41), (40, 40), (41, 38)])
def test_synthetic_star(shape):
    cutout, truth, middle = _star(shape)
    psf = _build(cutout, np.ones(shape, dtype=bool))

    assert psf.kernel.shape == shape
    assert psf.kernel.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.abs(_centroid(psf.kernel) - middle) < 0.01)
    assert psf.background == pytest.approx(BACKGROUND, abs=1e-6)
    np.testing.assert_allclose(psf.kernel, truth, atol=1e-4 * truth.max())
    np.testing.assert_allclose(
        psf.star_position,
        (ORIGIN[0] + middle[1] + OFFSET[1], ORIGIN[1] + middle[0] + OFFSET[0]),
        atol=0.01,
    )
    assert psf.exposure == "n0042"
    assert psf.pixel_scale == 0.009942
    assert psf.sampling == "detector"
    assert psf.pixel_response_included


@pytest.mark.parametrize("shape", [(41, 41), (40, 40)])
def test_masked_pixels_do_not_bias(shape):
    cutout, truth, middle = _star(shape)
    mask = np.ones(shape, dtype=bool)
    cy, cx = np.round(middle).astype(int)
    # A bad block on one side of the core, and most of the background frame
    # (so a median that ignored the mask would see the bad values), one NaN.
    mask[cy - 1 : cy + 1, cx + 2 : cx + 4] = False
    mask[:2] = mask[-2:] = mask[:, :2] = False
    cutout = cutout.copy()
    cutout[~mask] = 1e6
    cutout[0, 5] = np.nan

    psf = _build(cutout, mask)

    assert np.all(np.abs(_centroid(psf.kernel) - middle) < 0.01)
    assert psf.background == pytest.approx(BACKGROUND, abs=1e-6)
    assert psf.kernel.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(psf.kernel, truth, atol=1e-3 * truth.max())


@pytest.mark.parametrize("seed", range(5))
def test_noisy_star_centre(seed):
    shape = (41, 41)
    middle = (np.array(shape) - 1) / 2
    truth = _psf(shape, middle + OFFSET)
    # Peak signal-to-noise about 1000.
    noise = np.random.default_rng(seed).normal(0, 10, shape)
    cutout = 1e4 / truth.max() * truth + 50 + noise

    psf = _build(cutout, np.ones(shape, dtype=bool))

    expected = (ORIGIN[0] + middle[1] + OFFSET[1], ORIGIN[1] + middle[0] + OFFSET[0])
    np.testing.assert_allclose(psf.star_position, expected, atol=0.01)


def test_bit_identical():
    cutout, _, _ = _star((41, 40))
    mask = np.ones(cutout.shape, dtype=bool)
    mask[20, 22] = False
    first = _build(cutout, mask)
    second = _build(np.asfortranarray(cutout), np.asfortranarray(mask))
    assert first.kernel.tobytes() == second.kernel.tobytes()
    assert first.background == second.background
    assert first.star_position == second.star_position


@pytest.mark.parametrize("size", [41, 40])
def test_point_source_lands_on_its_pixel_through_psf_static(size):
    cutout, _, _ = _star((size, size))
    psf = _build(cutout, np.ones(cutout.shape, dtype=bool))

    field = np.zeros((64, 64))
    field[30, 27] = 1.0
    image = PsfStatic(jft.ShapeWithDtype(field.shape, float), psf.kernel)((field, {}))

    np.testing.assert_allclose(_centroid(np.asarray(image)), (30, 27), atol=0.01)


def test_prepared_fits_kernel(tmp_path):
    shape = (33, 34)
    middle = (np.array(shape) - 1) / 2
    path = tmp_path / "kernel.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(5 * _psf(shape, middle + OFFSET), name="PSF")]
    ).writeto(path)

    psf = kernel_from_fits(
        path, exposure="n0042", pixel_scale=0.009942, extension="PSF"
    )

    truth = _psf(shape, middle)
    assert psf.kernel.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.abs(_centroid(psf.kernel) - middle) < 0.01)
    np.testing.assert_allclose(psf.kernel, truth / truth.sum(), atol=1e-4 * truth.max())
    assert psf.background == 0.0
    assert psf.star_position is None


def test_rejects_bad_input():
    cutout, _, _ = _star((21, 21))
    good = np.ones(cutout.shape, dtype=bool)
    with pytest.raises(ValueError, match="finite"):
        _build(np.where(good, np.nan, cutout), good)
    with pytest.raises(ValueError, match="frame"):
        frameless = np.zeros(cutout.shape, dtype=bool)
        frameless[5:-5, 5:-5] = True
        _build(cutout, frameless)
    with pytest.raises(ValueError, match="positive flux"):
        _build(-cutout, good)
