import numpy as np

from ..color import Color
from astropy import units as u


ENERGY_UNIT_KEY = "energy_unit"
ENERGY_BIN_KEY = "energy_bin"
REFERENCE_BIN_KEY = "reference_bin"
EMIN_KEY = "e_min"
EMAX_KEY = "e_max"

DEFAULT_UNIT = u.Hz


def yaml_to_binned_colors(grid_config: dict) -> Color:
    """Convert YAML configuration to binned Color object for energy ranges.

    Parses a configuration dictionary (typically from YAML) containing energy
    bin specifications and converts them into a Color object representing
    spectral energy ranges. If no energy bins are specified, returns a default
    Color covering the full frequency range.

    Parameters
    ----------
    grid_config : dict
        Configuration dictionary containing energy bin specifications.
        Expected keys:

        - 'energy_unit' : str
            Name of the astropy unit for energy (e.g., 'keV', 'eV', 'Hz').
            Must be a valid attribute of astropy.units.
        - 'energy_bin' : list, array-like, or dict
            Energy bin specifications in one of two formats:

            1. Direct format: List or array of energy values or ranges
               e.g., [1.0, 2.0, 5.0] or [[1.0, 2.0], [3.0, 5.0]]

            2. Dictionary format with separate min/max arrays:
               {'e_min': [1.0, 3.0], 'e_max': [2.0, 5.0]}

    Returns
    -------
    Color
        A Color object representing the energy bins specified in the
        configuration. If 'energy_bin' key is not present, returns
        Color([0, np.inf] * u.Hz) as a default covering all frequencies.

    Notes
    -----
    The function handles two input formats for flexibility:

    1. **Sequential bins**: When energy_bin is a simple list like [1, 2, 5],
       it creates consecutive bins: [1-2], [2-5]

    2. **Explicit ranges**: When using the dictionary format with 'e_min'
       and 'e_max' keys, it creates potentially discontinuous ranges by
       pairing corresponding min/max values.

    The returned Color object automatically handles unit conversions between
    energy, frequency, and wavelength representations.

    Raises
    ------
    AttributeError
        If the specified 'energy_unit' is not a valid astropy.units attribute.
    KeyError
        If 'energy_bin' is a dict but missing 'e_min' or 'e_max' keys.
    TypeError
        If grid_config is not a dictionary.

    See Also
    --------
    Color : The class representing spectral color ranges.
    astropy.units : Astropy units framework for handling physical quantities.
    """

    if ENERGY_BIN_KEY not in grid_config:
        return Color([0, np.inf] * DEFAULT_UNIT)

    eunit = getattr(u, grid_config[ENERGY_UNIT_KEY])
    energy_bin = grid_config[ENERGY_BIN_KEY]

    if isinstance(energy_bin, dict):
        emins = energy_bin[EMIN_KEY]
        emaxs = energy_bin[EMAX_KEY]
        energy_bin = [(emin, emax) for emin, emax in zip(emins, emaxs)]

    return Color(energy_bin * eunit)


def cfg_to_binned_colors(grid_config: dict) -> Color:
    if not grid_config.get("frequencies"):
        return Color([0, np.inf] * DEFAULT_UNIT)

    frequencies = map(float, grid_config["frequencies"].split(","))
    frequencies = np.sort(np.array(list(frequencies)))

    color_ranges = []
    for ii in range(len(frequencies) - 1):
        emin = frequencies[ii]
        emax = frequencies[ii + 1]
        color_ranges.append([emin, emax])

    return Color(color_ranges * DEFAULT_UNIT)
