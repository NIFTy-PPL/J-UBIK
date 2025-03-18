# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from typing import Tuple, Optional

from astropy import units
from astropy.coordinates import SkyCoord

from .reconstruction_grid import Grid


def _get_world_location(config: dict) -> SkyCoord:
    """Get the world location from the config."""
    ra = config["grid"]["pointing"]["ra"]
    dec = config["grid"]["pointing"]["dec"]
    frame = config["grid"]["pointing"].get("frame", "icrs")
    unit = getattr(units, config["grid"]["pointing"].get("unit", "deg"))
    return SkyCoord(ra=ra * unit, dec=dec * unit, frame=frame)


def _get_shape(config: dict) -> Tuple[int, int]:
    """Get the shape from the config."""
    npix = config["grid"]["sdim"]
    return (npix, npix)


def _get_fov(config: dict) -> Tuple[units.Quantity, units.Quantity]:
    """Get the fov from the config."""
    fov = config["grid"]["fov"]
    unit = getattr(units, config["grid"].get("fov_unit", "arcsec"))
    return fov * unit


def _get_rotation(config: dict) -> units.Quantity:
    """Get the rotation from the config."""
    rotation = config["grid"]["pointing"]["rotation"]
    unit = getattr(units, config["grid"]["pointing"].get("unit", "deg"))
    return rotation * unit


def build_reconstruction_grid_from_config(config: dict) -> Grid:
    """
    Builds the reconstruction grid from the given configuration.

    The reconstruction grid is defined by the world location, field of view
    (FOV), shape (resolution), and rotation, all specified in the input
    configuration. These parameters are extracted from the config dictionary
    using helper functions.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing the following keys:
        - 'world_location': World coordinates defining the center of the grid.
        - 'fov': Field of view of the grid in appropriate units.
        - 'shape': Shape of the grid (resolution) as a tuple (nx, ny).
        - 'rotation': Rotation of the grid in degrees.

    Returns
    -------
    Grid
        A `Grid` object constructed using the world location, shape,
        field of view, and rotation provided in the configuration.
    """
    wl = _get_world_location(config)
    fov = _get_fov(config)
    shape = _get_shape(config)
    rotation = _get_rotation(config)
    return Grid(wl, shape, (fov.to(units.deg), fov.to(units.deg)), rotation=rotation)


def build_filter_zero_flux(
    config: dict,
    filter: str,
) -> dict:
    """
    Builds the zero flux prior for the specified filter.

    This function retrieves the zero flux prior for a given filter from the
    configuration dictionary. If the filter-specific prior is not available,
    it falls back to a general prior defined in the configuration.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing the zero flux priors for
        different filters under `config['telescope']['zero_flux']`.

    filter : str
        The name of the filter (case-insensitive) for which the zero flux prior
        is to be built.

    Returns
    -------
    dict
        A dictionary containing the zero flux prior for the specified filter.
        If the filter is not present, returns the default prior.
    """
    prior_config = config["telescope"]["zero_flux"]
    lower_filter = filter.lower()

    if lower_filter in prior_config:
        return dict(prior=prior_config[lower_filter])

    return dict(prior=prior_config["prior"])


def build_coordinates_correction_prior_from_config(
    config: dict, filter: Optional[str] = "", filter_data_set_id: Optional[int] = 0
) -> dict:
    """
    Builds the coordinate correction prior for the specified filter and dataset.

    The function extracts the shift and rotation priors for the given filter and
    dataset ID from the configuration.
    If the specific filter or dataset ID is not found, it returns the default
    shift and rotation priors. The rotation prior is converted to radians
    if needed.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing rotation and shift priors under
        `config['telescope']['rotation_and_shift']['priors']`.

    filter : Optional[str], default=''
        The name of the filter (case-insensitive) for which the prior is needed.
        If not specified, the default priors are used.

    filter_data_set_id : Optional[int], default=0
        The dataset ID for which the prior is needed. If not provided or not
        found, the function uses the default dataset priors.

    Returns
    -------
    dict
        A dictionary containing the shift and rotation priors for the specified
        filter and dataset. If the filter or dataset is not found, returns the
        default priors.
    """
    rs_priors = config["telescope"]["rotation_and_shift"]["priors"]

    lower_filter = filter.lower()
    if (lower_filter in rs_priors) and (
        filter_data_set_id in rs_priors.get(lower_filter, dict())
    ):
        shift = rs_priors[lower_filter][filter_data_set_id]["shift"]
        rotation = rs_priors[lower_filter][filter_data_set_id]["rotation"]

    else:
        shift = rs_priors["shift"]
        rotation = rs_priors["rotation"]

    rotation_unit = getattr(units, rs_priors.get("rotation_unit", "deg"))
    rotation = (
        rotation[0],
        rotation[1],
        (rotation[2] * rotation_unit).to(units.rad).value,
    )
    return dict(shift=shift, rotation=rotation)


def config_transform(config: dict):
    """
    Recursively transforms string values in a configuration dictionary.

    This function processes a dictionary and attempts to evaluate any string
    values that may represent valid Python expressions. If the string cannot
    be evaluated, it is left unchanged. The function also applies the same
    transformation recursively for any nested dictionaries.

    Parameters
    ----------
    config : dict
        The configuration dictionary where string values may be transformed.
        If a value is a string that can be evaluated, it is replaced by the
        result of `eval(val)`. Nested dictionaries are processed recursively.
    """
    for key, val in config.items():
        if isinstance(val, str):
            try:
                config[key] = eval(val)
            except:
                continue
        elif isinstance(val, dict):
            config_transform(val)
