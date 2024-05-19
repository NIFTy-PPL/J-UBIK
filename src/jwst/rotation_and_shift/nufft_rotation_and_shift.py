from jax_finufft import nufft2
from jax.numpy.fft import ifftshift, ifft2
from jax.numpy import reshape

from numpy import pi, array

from numpy.typing import ArrayLike
from typing import Tuple, Callable, Optional


def build_nufft_rotation_and_shift(
    sky_dvol: ArrayLike,
    sub_dvol: ArrayLike,
    subsample_centers: ArrayLike,
    sky_shape: Tuple[int, int],
    sky_as_brightness: bool = False
) -> Callable[ArrayLike, ArrayLike]:
    '''Building nuFFT interpolation model.

    Parameters
    ----------
    sky_dvol : float
        The volume of the sky/reconstruction pixels

    sub_dvol : float
        The volume of the subsample pixels.
        Typically the data pixel is subsampled

    subsample_centers : array
        Coordinates of the subsample centers in the reconstruction pixel frame

    mask : array
        Mask of the data array

    sky_shape : tuple
        The shape of the reconstruction array (sky shape)

    sky_as_brightness : bool (default False)
        If True, the sky will be treated as a brightness distribution.
        This is the same as setting sky_dvol = 1.

    Returns
    -------
    rotate_shift_subsample : function
        The interpolation function

    Notes
    -----
    The sky is the reconstruction array, we assume a one to one relation
    between the sky brightness and the flux:
        flux(x, y) = sky(x, y) * sky_dvol

    '''

    out_shape = subsample_centers.shape[1:]

    # The conversion factor from sky to subpixel
    # (flux = sky_brightness * flux_conversion)
    if sky_as_brightness:
        sky_dvol = 1
    flux_conversion = sub_dvol / sky_dvol

    subsample_centers = subsample_centers.reshape(2, -1)
    xy_finufft = (
        2 * pi * subsample_centers /
        array(sky_shape)[:, None]
    )

    def rotate_shift_subsample(field, correction):
        f_field = ifftshift(ifft2(field))
        out = nufft2(f_field, xy_finufft[0], xy_finufft[1]).real
        return reshape(out, out_shape) * flux_conversion

    return rotate_shift_subsample
