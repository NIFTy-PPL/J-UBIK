"""Empirical NIRC2 PSF kernels measured from PSF stars.

The kernels are effective PSFs: sampled on the detector pixels, so they already
contain the pixel response. Each kernel records the exposure it was measured in;
which science exposures it is used for is the caller's explicit choice.
"""

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike
from scipy.ndimage import fourier_shift

_MAX_ITERATIONS = 100
_CENTRE_TOLERANCE = 1e-6  # pixels


@dataclass(frozen=True)
class EmpiricalPsfKernel:
    """A unit-sum, detector-sampled PSF kernel and where it was measured.

    The star centre (its windowed centroid) sits at `(K - 1) / 2` on each axis,
    the centre `PsfStatic` expects for odd and even `K`.

    Attributes
    ----------
    kernel: np.ndarray
        The 2-D kernel, summing to one.
    exposure: str
        The exposure the star was measured in.
    star_position: tuple[float, float] | None
        The 0-based detector pixel (x, y) of the star centre, None if unknown.
    background: float
        The background level subtracted before normalizing, in input units.
    pixel_scale: float
        The detector pixel scale in arcsec per pixel.
    sampling: str
        Always "detector": one kernel pixel is one detector pixel.
    pixel_response_included: bool
        Always True: the kernel must not be integrated over pixels again.
    """

    kernel: np.ndarray
    exposure: str
    star_position: tuple[float, float] | None
    background: float
    pixel_scale: float
    sampling: str = "detector"
    pixel_response_included: bool = True


def _shift(image: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """`image` moved by `shift` (y, x) pixels, periodically, with a Fourier phase."""
    return np.fft.ifft2(fourier_shift(np.fft.fft2(image), shift)).real


def _centroid(image: np.ndarray, around: np.ndarray) -> np.ndarray:
    """First moment (y, x) of `image` under a Gaussian window centred on `around`.

    The window keeps noise in the far wings, which a plain first moment weights
    most, from moving the centre. For a point-symmetric star the iterated
    windowed centroid is the plain one.
    """
    y, x = np.indices(image.shape)
    # shortcut: window width fixed at min(shape) / 8; make it a parameter if
    # cutouts much larger than the PSF core turn out noise-limited.
    width = min(image.shape) / 8
    r2 = (y - around[0]) ** 2 + (x - around[1]) ** 2
    weighted = image * np.exp(-0.5 * r2 / width**2)
    return np.array([np.sum(y * weighted), np.sum(x * weighted)]) / np.sum(weighted)


def _centred_unit_kernel(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel with the star centre moved to `(K - 1) / 2`, and the centre (y, x) it
    had in `image`.

    The centre is the windowed centroid. Masked pixels are filled from the point
    reflection of the image about it, iterated to a fixed point, so a mask that
    is not point-symmetric about the star does not pull the centre. This assumes
    a roughly point-symmetric PSF, and leaves the reflected values in the kernel.
    """
    if np.sum(image[mask]) <= 0:
        raise ValueError("The star has no positive flux after background subtraction.")
    middle = (np.array(image.shape) - 1) / 2
    filled = np.where(mask, image, 0.0)
    centre = _centroid(filled, middle)
    for _ in range(_MAX_ITERATIONS):
        # The flip maps pixel p to 2 * middle - p, the shift then onto 2 * centre - p.
        reflected = _shift(filled[::-1, ::-1], 2 * (centre - middle))
        filled = np.where(mask, image, reflected)
        previous, centre = centre, _centroid(filled, centre)
        if np.all(np.abs(centre - previous) < _CENTRE_TOLERANCE):
            break
    else:
        raise ValueError(
            "The star centroid does not converge; too much of the star is masked."
        )
    kernel = _shift(filled, middle - centre)
    return kernel / np.sum(kernel), centre


def kernel_from_star_cutout(
    cutout: ArrayLike,
    mask: ArrayLike,
    *,
    exposure: str,
    origin: tuple[int, int],
    pixel_scale: float,
    border: int = 2,
) -> EmpiricalPsfKernel:
    """Empirical PSF kernel from a cutout around an isolated star.

    The background is the median of the unmasked pixels in the outer `border`
    rows and columns, so the cutout should be large enough that the star's
    wings are negligible there. The kernel has the shape of the cutout. The
    centring shift is periodic, so the star should sit near the cutout centre.
    Masked pixels take the value of their point reflection through the star.
    Where that is masked too, as in a masked core, the kernel there is only an
    interpolation, so such stars give approximate kernels.

    Parameters
    ----------
    cutout: ArrayLike
        The 2-D star cutout, in any linear flux unit.
    mask: ArrayLike
        Boolean, True where the cutout pixel is valid. Masked pixels may hold
        any value, including NaN.
    exposure: str
        The exposure the cutout is taken from.
    origin: tuple[int, int]
        The 0-based detector pixel (x, y) of `cutout[0, 0]`.
    pixel_scale: float
        The detector pixel scale in arcsec per pixel.
    border: int
        The width in pixels of the background frame.
    """
    # C order, so the same pixel values give the same bits whatever the layout.
    cutout = np.ascontiguousarray(cutout, dtype=float)
    mask = np.ascontiguousarray(mask, dtype=bool)
    if cutout.ndim != 2 or mask.shape != cutout.shape:
        raise ValueError(
            f"Need a 2-D cutout and a mask of its shape, got {cutout.shape} and "
            f"{mask.shape}."
        )
    if not np.all(np.isfinite(cutout[mask])):
        raise ValueError("Unmasked cutout pixels must be finite.")
    if not 0 < border < min(cutout.shape) / 2:
        raise ValueError(f"`border` {border} does not fit the cutout {cutout.shape}.")

    frame = np.ones(cutout.shape, dtype=bool)
    frame[border:-border, border:-border] = False
    if not np.any(frame & mask):
        raise ValueError("No unmasked pixel in the background frame.")
    background = float(np.median(cutout[frame & mask]))

    kernel, (cy, cx) = _centred_unit_kernel(cutout - background, mask)
    return EmpiricalPsfKernel(
        kernel=kernel,
        exposure=exposure,
        star_position=(float(origin[0] + cx), float(origin[1] + cy)),
        background=background,
        pixel_scale=pixel_scale,
    )


def kernel_from_fits(
    path: str,
    *,
    exposure: str,
    pixel_scale: float,
    extension: int | str = 0,
    star_position: tuple[float, float] | None = None,
) -> EmpiricalPsfKernel:
    """Empirical PSF kernel from a prepared, background-free kernel FITS file.

    The kernel is recentred on its centre and normalized to unit sum like a
    star cutout, so it follows the same centre convention whatever the file
    used. No background is subtracted.

    Parameters
    ----------
    path: str
        The FITS file.
    exposure: str
        The exposure the kernel was measured in.
    pixel_scale: float
        The detector pixel scale in arcsec per pixel.
    extension: int | str
        The HDU index or name holding the 2-D kernel.
    star_position: tuple[float, float] | None
        The 0-based detector pixel (x, y) of the star, if known.
    """
    kernel = np.asarray(fits.getdata(path, extension), dtype=float)
    if kernel.ndim != 2 or not np.all(np.isfinite(kernel)):
        raise ValueError(
            f"{path}[{extension}] must hold a finite 2-D kernel, got shape "
            f"{kernel.shape}."
        )
    kernel, _ = _centred_unit_kernel(kernel, np.ones(kernel.shape, dtype=bool))
    return EmpiricalPsfKernel(
        kernel=kernel,
        exposure=exposure,
        star_position=star_position,
        background=0.0,
        pixel_scale=pixel_scale,
    )
