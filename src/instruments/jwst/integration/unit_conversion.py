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
    """Builds a flux conversion between to grids.

    Parameters
    ----------
    sky_dvol: float
        The volume of the sky/reconstruction pixels
    sub_dvol: float
        The volume of the subsample pixels.
        Typically, the data pixel is subsampled.
    """

    _check_physical_types([sky_unit, data_unit])

    if sky_unit.physical_type == sky_unit.physical_type:
        return _build_same_physical_type(sky_unit, data_unit)

    return _build_different_physical_type(sky_unit, sky_dvol, data_unit, data_dvol)


def _build_same_physical_type(
    sky_unit: u.Unit,
    data_unit: u.Unit,
):
    if sky_unit == data_unit:
        return lambda x: x

    conversion = sky_unit.to(data_unit)
    return lambda x: x * conversion


def _build_different_physical_type(
    sky_unit: u.Unit,
    sky_dvol: u.Quantity,
    data_unit: u.Unit,
    data_dvol: u.Quantity,
):
    if sky_unit.physical_type in SURFACE_BRIGHTNESS:
        tsky_unit = sky_unit * sky_dvol
        conversion = tsky_unit.to(data_unit) * sky_dvol.value
        assert isinstance(conversion, float)
        return lambda x: x * conversion

    elif sky_unit.physical_type in FLUX:
        tsky_unit = sky_unit / sky_dvol
        conversion = tsky_unit.to(data_unit) / sky_dvol.value
        assert isinstance(conversion, float)
        return lambda x: x * conversion

    else:
        raise ValueError(
            f"Sky unit's physical type not implemented: sky_unit={sky_unit}."
        )
