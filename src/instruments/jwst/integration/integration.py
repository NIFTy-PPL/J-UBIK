from astropy import units as u
from .unit_conversion import FLUX, SURFACE_BRIGHTNESS


def integration_factory(
    unit: u.Unit,
    high_resolution_shape: tuple,
    reduction_factor: int,
):
    """
    Builds a function that reduces a grid by a reduction factor. When the field is in
    flux units, the field will be summed. When the field is surface brightness, the
    average of the field is computed.

    Parameters
    ----------
    unit: astropy.units.Unit
        The unit of the field.
    high_resolution_shape : tuple of int
        The shape of the high-resolution grid (height, width).
    reduction_factor : int
        The factor by which to reduce each dimension of the grid.
        It must evenly divide both the height and width of the input grid.

    Returns
    -------
    callable
        A function that, when applied to an array of shape `high_resolution_shape`,
        reduces its resolution by the specified reduction factor.

    Raises
    ------
    ValueError
        If the `reduction_factor` does not evenly divide both dimensions
        of `high_resolution_shape`. Or the unit of the field is not implemented.

    Example
    -------
    Given an input shape of (100, 100) and a reduction factor of 10,
    the resulting callable reshapes the input into blocks of shape (10, 10)
    and sums/averages them, reducing the resolution to (10, 10).
    """
    if (high_resolution_shape[0] % reduction_factor != 0) or (
        high_resolution_shape[1] % reduction_factor != 0
    ):
        raise ValueError("The reduction factor must evenly divide both dimensions")

    new_shape = (
        high_resolution_shape[0] // reduction_factor,
        reduction_factor,
        high_resolution_shape[1] // reduction_factor,
        reduction_factor,
    )

    if unit.physical_type in FLUX:
        return lambda x: x.reshape(new_shape).sum(axis=(1, 3))

    elif unit.physical_type in SURFACE_BRIGHTNESS:
        return lambda x: x.reshape(new_shape).mean(axis=(1, 3))

    else:
        raise ValueError(f"Unit not implemented: {unit}")
