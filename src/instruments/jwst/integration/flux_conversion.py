def build_flux_conversion(
    sky_dvol: float,
    sub_dvol: float,
    as_brightness: bool = True,
    sky_as_brightness: bool = False,
):
    """Builds a flux conversion between to grids.

    Parameters
    ----------
    sky_dvol: float
        The volume of the sky/reconstruction pixels
    sub_dvol: float
        The volume of the subsample pixels.
        Typically, the data pixel is subsampled.
    """

    if as_brightness:
        return lambda x: x

    assert reconstruction_grid.spatial.dvol.unit == data_grid_dvol.unit

    # The conversion factor from sky to subpixel
    # (flux = sky_brightness * flux_conversion)
    if sky_as_brightness:
        sky_dvol = 1
    flux_conversion = sub_dvol / sky_dvol

    return lambda x: x * flux_conversion
