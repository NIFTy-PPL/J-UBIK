from astropy import units as u

SURFACE_BRIGHTNESS = {u.physical.surface_brightness}
FLUX = {u.physical.spectral_flux_density, u.physical.energy_density}


def _check_physical_types(units: list[u.Unit]):
    for unit in units:
        assert unit.physical_type in FLUX | SURFACE_BRIGHTNESS, (
            f"Unit not implemented: {unit}"
        )


def build_unit_conversion(
    sky_unit: u.Unit,
    sky_dvol: u.Quantity,
    data_unit: u.Unit,
    data_dvol: u.Quantity,
):
    """Converts sky values sampled on the subsampled data grid from `sky_unit` to
    `data_unit`; flux units are flux per pixel of `sky_dvol` or `data_dvol` (the
    solid angle of a sky pixel and of a subsampled data pixel, respectively).

    Parameters
    ----------
    sky_unit: astropy.units.Unit
        Unit of the sky, surface brightness or flux per sky pixel.
    sky_dvol: astropy.units.Quantity
        Solid angle of a sky pixel.
    data_unit: astropy.units.Unit
        Unit of the data, surface brightness or flux per data pixel.
    data_dvol: astropy.units.Quantity
        Solid angle of a subsampled data pixel.
    """

    _check_physical_types([sky_unit, data_unit])

    # Convert to surface brightness on the sky grid, then to flux per subpixel.
    scale = 1.0 * sky_unit
    if sky_unit.physical_type in FLUX:
        scale = scale / sky_dvol
    if data_unit.physical_type in FLUX:
        scale = scale * data_dvol
    conversion = scale.to_value(data_unit)
    return lambda x: x * conversion
