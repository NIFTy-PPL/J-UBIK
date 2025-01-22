from ...likelihood import (build_gaussian_likelihood)
from .config_handler import (get_grid_extension_from_config,)

from .jwst_response import build_jwst_response
from .jwst_data import (load_jwst_data_mask_std)
from .jwst_data import JWST_FILTERS
from .color import Color, ColorRange
from .grid import Grid
from .filter_projector import (
    FilterProjector)

# Parsing
from .parse.jwst_psf import (yaml_to_psf_kernel_config)
from .parse.zero_flux_model import (yaml_to_zero_flux_prior_config)
from .parse.rotation_and_shift.coordinates_correction import (
    yaml_to_coordinates_correction_config)
from .parse.rotation_and_shift.rotation_and_shift import (
    yaml_to_rotation_and_shift_algorithm_config)
from .jwst_psf import load_psf_kernel_from_config

import jax.numpy as jnp
import nifty8.re as jft

# std
from functools import reduce
from astropy import units as u
from typing import Union, Optional


def build_filter_projector(
    sky_model: jft.Model,
    grid: Grid,
    data_filter_names: list[str],
    sky_key: str = 'sky',
) -> FilterProjector:
    named_color_ranges = {}
    for name, values in JWST_FILTERS.items():
        pivot, bw, er, blue, red = values
        named_color_ranges[name] = ColorRange(
            Color(red*u.um), Color(blue*u.um))

    keys_and_colors = {}
    keys_and_index = {}
    for color_index, grid_color_range in enumerate(grid.spectral):
        for name in data_filter_names:
            jwst_filter = named_color_ranges[name.upper()]
            if grid_color_range.center in jwst_filter:
                keys_and_colors[name] = grid_color_range
                keys_and_index[name] = color_index

    filter_projector = FilterProjector(
        sky_domain=sky_model.target,
        keys_and_colors=keys_and_colors,
        keys_and_index=keys_and_index,
        sky_key=sky_key,
    )

    for fpt, fpc in filter_projector.target.items():
        print(fpt, fpc)

    return filter_projector


def build_jwst_likelihoods(
    cfg: dict,
    grid: Grid,
    sky_model: jft.Model,
    sky_key: str = 'sky',
    files_key: str = 'files',
    telescope_key: str = 'telescope',
    sky_unit: Optional[u.Unit] = None,
) -> Union[jft.Likelihood, FilterProjector, dict]:
    '''Build the jwst likelihood according to the config and grid.'''

    filter_projector = build_filter_projector(
        sky_model, grid, cfg[files_key]['filter'].keys())

    # Parsing
    zero_flux_prior_configs = yaml_to_zero_flux_prior_config(
        cfg[telescope_key]['zero_flux'])
    psf_kernel_configs = yaml_to_psf_kernel_config(cfg[telescope_key]['psf'])
    coordiantes_correction_config = yaml_to_coordinates_correction_config(
        cfg[telescope_key]['rotation_and_shift']['correction_priors'])
    rotation_and_shift_algorithm_config = yaml_to_rotation_and_shift_algorithm_config(
        cfg[telescope_key]['rotation_and_shift'])

    data_dict = {}
    likelihoods = []
    for fltname, flt in cfg[files_key]['filter'].items():
        for ii, filepath in enumerate(flt):
            print(fltname, ii, filepath)

            # Loading data, std, and mask.
            grid_extension = get_grid_extension_from_config(
                cfg[telescope_key], grid)
            world_corners = grid.spatial.world_extrema(ext=grid_extension)

            jwst_data, data, mask, std = load_jwst_data_mask_std(
                filepath, grid, world_corners)
            if sky_unit is not None:
                assert jwst_data.dm.meta.bunit_data == sky_unit

            data_subsample = cfg[telescope_key]['rotation_and_shift']['subsample']
            psf_kernel = load_psf_kernel_from_config(
                jwst_data=jwst_data,
                pointing_center=grid.spatial.center,
                subsample=data_subsample,
                config_parameters=psf_kernel_configs,
            )

            energy_name = filter_projector.get_key(jwst_data.pivot_wavelength)
            data_identifier = f'{fltname}_{ii}'

            jwst_response = build_jwst_response(
                sky_domain={energy_name: filter_projector.target[energy_name]},
                data_identifier=data_identifier,
                data_subsample=data_subsample,

                rotation_and_shift_kwargs=dict(
                    reconstruction_grid=grid,
                    data_dvol=jwst_data.dvol,
                    data_wcs=jwst_data.wcs,
                    algorithm_config=rotation_and_shift_algorithm_config,
                    world_extrema=world_corners,
                ),
                shift_and_rotation_correction_prior=coordiantes_correction_config.get_filter_or_default(
                    jwst_data.filter, ii),

                psf_kernel=psf_kernel,
                transmission=jwst_data.transmission,
                zero_flux_prior_config=zero_flux_prior_configs.get_filter_or_default(
                    fltname),
                data_mask=mask,
            )

            data_dict[data_identifier] = dict(
                index=filter_projector.keys_and_index[energy_name],
                data=data,
                std=std,
                mask=mask,
                data_model=jwst_response,
                data_dvol=jwst_data.dvol,
                data_transmission=jwst_data.transmission,
            )

            likelihood = build_gaussian_likelihood(
                jnp.array(data[mask], dtype=float),
                jnp.array(std[mask], dtype=float))
            likelihood = likelihood.amend(
                jwst_response, domain=jft.Vector(jwst_response.domain))
            likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x+y, likelihoods)
    return likelihood, filter_projector, data_dict
