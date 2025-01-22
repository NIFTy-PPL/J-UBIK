import numpy as np

from ..color import ColorRanges, ColorRange, Color
from astropy import units as u


ENERGY_UNIT_KEY = 'energy_unit'
ENERGY_BIN_KEY = 'energy_bin'
REFERENCE_BIN_KEY = 'reference_bin'

CFG_REFERENCE_BIN_KEY = 'frequency reference bin'


def yaml_to_binned_colors(grid_config: dict) -> ColorRanges:
    EMIN_KEY = 'e_min'
    EMAX_KEY = 'e_max'

    color_ranges = []
    emins = grid_config[ENERGY_BIN_KEY][EMIN_KEY]
    emaxs = grid_config[ENERGY_BIN_KEY][EMAX_KEY]
    shape = grid_config[ENERGY_BIN_KEY].get('shape')
    eunit = getattr(u, grid_config[ENERGY_UNIT_KEY])

    if isinstance(emins, float) and isinstance(emins, float) and shape is not None:
        ebins = np.linspace(emins, emaxs, shape+1)
        emins = ebins[:-1]
        emaxs = ebins[1:]

    for emin, emax in zip(emins, emaxs):
        emin, emax = emin*eunit, emax*eunit
        color_ranges.append(ColorRange(Color(emin), Color(emax)))

    return ColorRanges(color_ranges)


def yaml_to_color_reference_bin(grid_config: dict) -> int:
    return grid_config[ENERGY_BIN_KEY][REFERENCE_BIN_KEY]


def cfg_to_binned_colors(grid_config: dict) -> ColorRanges:
    frequency_unit = u.Hz

    if not grid_config.get("frequencies"):
        emin = -np.inf*frequency_unit
        emax = np.inf*frequency_unit
        return ColorRanges([ColorRange(Color(emin), Color(emax))])

    frequencies = map(float, grid_config["frequencies"].split(","))
    frequencies = np.sort(np.array(list(frequencies)))

    color_ranges = []
    for ii in range(len(frequencies)-1):
        emin = frequencies[ii]*frequency_unit
        emax = frequencies[ii+1]*frequency_unit
        color_ranges.append(
            ColorRange(Color(emin), Color(emax))
        )

    return ColorRanges(color_ranges)


def cfg_to_color_reference_bin(grid_config: dict) -> int:
    return grid_config.get(CFG_REFERENCE_BIN_KEY, 0)
