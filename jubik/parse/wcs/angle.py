from astropy import units as u
from astropy.coordinates import Angle


def parse_angle(config: dict, key: str, default: u.Quantity) -> u.Quantity:
    """Parse an angular entry from the config.

    The value is read with `astropy.coordinates.Angle`, which accepts strings
    such as `12uas` or `12h30m10s`, and which rejects values without unit as
    well as values carrying a non-angular unit.

    Parameters
    ----------
    config : dict
        Configuration holding the angle under `key`.
    key : str
        Name of the configuration entry.
    default : u.Quantity
        Value used when `key` is not part of `config`.

    Returns
    -------
    u.Quantity
        The parsed angle.

    Raises
    ------
    ValueError
        If the entry cannot be read as an angle, i.e. when it carries no unit
        or a unit which is not equivalent to `rad`.
    """

    value = config.get(key, default)

    try:
        return Angle(value)
    except (u.UnitsError, TypeError, ValueError) as e:
        raise ValueError(
            f'`{key}` should be an angle carrying an angular unit, '
            f'got {value!r}.'
        ) from e
