from ...likelihood import build_gaussian_likelihood
from .config_handler import (
    get_grid_extension_from_config,
)

from .jwst_response import build_jwst_response
from .jwst_data import JwstData
from ...grid import Grid
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
)
from .jwst_psf import load_psf_kernel

# Parsing
from .parse.jwst_psf import yaml_to_psf_kernel_config
from .parse.zero_flux_model import yaml_to_zero_flux_prior_config
from .parse.rotation_and_shift.coordinates_correction import (
    yaml_to_coordinates_correction_config,
)
from .parse.rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
)
from .parse.jwst_response import SkyMetaInformation
from .parse.masking.data_mask import yaml_to_corner_mask_configs
from .plotting.residuals import ResidualPlottingInformation


# Libraries
import jax.numpy as jnp
import nifty8.re as jft
from nifty8.logger import logger

# std
from functools import reduce
from astropy import units as u
from typing import Union


def build_jwst_likelihoods(
    cfg: dict,
    grid: Grid,
    sky_model: jft.Model,
    sky_key: str = "sky",
    files_key: str = "files",
    telescope_key: str = "telescope",
    sky_unit: u.Unit | None = None,
) -> Union[jft.Likelihood, FilterProjector, dict]:
    """Build the jwst likelihood according to the config and grid."""

    filter_projector = build_filter_projector(
        sky_model, grid, cfg[files_key]["filter"].keys()
    )

    # Parsing
    zero_flux_prior_configs = yaml_to_zero_flux_prior_config(
        cfg[telescope_key]["zero_flux"]
    )
    psf_kernel_configs = yaml_to_psf_kernel_config(cfg[telescope_key]["psf"])
    coordiantes_correction_config = yaml_to_coordinates_correction_config(
        cfg[telescope_key]["rotation_and_shift"]["correction_priors"]
    )
    rotation_and_shift_algorithm = rotation_and_shift_algorithm_config_factory(
        cfg[telescope_key]["rotation_and_shift"]
    )
    data_subsample = cfg[telescope_key]["rotation_and_shift"]["subsample"]
    sky_meta = SkyMetaInformation(
        grid_extension=get_grid_extension_from_config(cfg[telescope_key], grid),
        unit=sky_unit,
    )

    target_plotting = {}
    likelihoods = []
    for fltname, flt in cfg[files_key]["filter"].items():
        for ii, filepath in enumerate(flt):
            logger.info(f"Loading: {fltname} {ii} {filepath}")

            # Loading data, std, and mask.
            jwst_data = JwstData(
                filepath, identifier=f"{fltname}_{ii}", subsample=data_subsample
            )
            data, mask, std = jwst_data.bounding_data_mask_std_by_world_corners(
                grid.spatial,
                grid.spatial.world_corners(extension_value=sky_meta.grid_extension),
                yaml_to_corner_mask_configs(cfg[telescope_key]),
            )
            energy_name = filter_projector.get_key(jwst_data.pivot_wavelength)

            psf_kernel = load_psf_kernel(
                jwst_data=jwst_data,
                target_center=grid.spatial.center,
                config_parameters=psf_kernel_configs,
            )

            shift_and_rotation_correction = ShiftAndRotationCorrection(
                domain_key=f"{jwst_data.meta.identifier}_correction",
                correction_prior=coordiantes_correction_config.get_name_setting_or_default(
                    jwst_data.filter, ii
                ),
                rotation_center=jwst_data.get_boresight_world_coords(),
            )

            jwst_target_response = build_jwst_response(
                sky_domain={energy_name: filter_projector.target[energy_name]},
                data_wcs=jwst_data.wcs,
                data_meta=jwst_data.meta,
                sky_wcs=grid.spatial,
                sky_meta=sky_meta,
                rotation_and_shift_algorithm=rotation_and_shift_algorithm,
                shift_and_rotation_correction=shift_and_rotation_correction,
                psf_kernel=psf_kernel,
                transmission=jwst_data.transmission,
                zero_flux_prior_config=zero_flux_prior_configs.get_name_setting_or_default(
                    fltname
                ),
                data_mask=mask,
            )

            target_plotting[jwst_data.meta.identifier] = ResidualPlottingInformation(
                data=data,
                std=std,
                mask=mask,
                model=jwst_target_response,
            )

            likelihood = build_gaussian_likelihood(
                jnp.array(data[mask], dtype=float), jnp.array(std[mask], dtype=float)
            )
            likelihood = likelihood.amend(
                jwst_target_response, domain=jft.Vector(jwst_target_response.domain)
            )
            likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x + y, likelihoods)
    return likelihood, filter_projector, target_plotting
