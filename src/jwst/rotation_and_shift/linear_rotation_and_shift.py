from jax.scipy.ndimage import map_coordinates
from jax import vmap

from functools import partial

from numpy.typing import ArrayLike
from typing import Callable


def build_linear_rotation_and_shift(
    sky_dvol: float,
    sub_dvol: float,
    subsample_centers: ArrayLike,
    order: int = 1,
    updating: bool = False,
    sky_as_brightness: bool = False
) -> Callable[ArrayLike, ArrayLike]:
    '''Building linear (higher orders not yet supported) interpolation model.

    Parameters
    ----------
    sky_dvol : float
        The volume of the sky/reconstruction pixels

    sub_dvol : float
        The volume of the subsample pixels.
        Typically the data pixel is subsampled

    subsample_centers : array
        The coordinates of the subsample centers

    mask : array
        Mask of the data array

    order : int (default 1)
        The order of the interpolation scheme (only linear supported by JAX)

    updating : bool (default False)
        If True, a model for an xy_shift can be supplied which will infer the
        a linear shift correction.

    sky_as_brightness : bool (default False)
        If True, the sky will be treated as a brightness distribution.
        This is the same as setting sky_dvol = 1.

    Returns
    -------
    rotation_shift_subsample : function
        The interpolation function

    Notes
    -----
    The sky is the reconstruction array, we assume a one to one relation
    between the sky brightness and the flux:
        flux(x, y) = sky(x, y) * sky_dvol
    '''

    # The conversion factor from sky to subpixel
    # (flux = sky_brightness * flux_conversion)
    if sky_as_brightness:
        sky_dvol = 1
    flux_conversion = sub_dvol / sky_dvol

    interpolation = partial(
        map_coordinates, order=order, mode='wrap')

    if updating:
        def rotation_shift_subsample(x, y):
            field, xy_shift = x, y
            out = interpolation(
                field, subsample_centers - xy_shift[None, :, None])
            return out * flux_conversion

    else:
        def rotation_shift_subsample(x, y):
            out = interpolation(x, subsample_centers)
            return out * flux_conversion

    return rotation_shift_subsample
