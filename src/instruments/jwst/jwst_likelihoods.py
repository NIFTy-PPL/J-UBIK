from ...likelihood import build_gaussian_likelihood
from ...grid import Grid
from ...wcs.wcs_jwst_data import WcsJwstData
from .jwst_response import build_jwst_response
from .jwst_data import JwstData, DataMetaInformation
from .jwst_psf import load_psf_kernel
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
)
from .plotting.residuals import ResidualPlottingInformation
from .config_handler import (
    get_grid_extension_from_config,
)

# Parsing
from .parse.jwst_psf import yaml_to_psf_kernel_config
from .parse.zero_flux_model import yaml_to_zero_flux_prior_config
from .parse.rotation_and_shift.coordinates_correction import (
    yaml_to_coordinates_correction_config,
    CoordinatesCorrectionConfig,
)
from .parse.rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
)
from .parse.jwst_response import SkyMetaInformation
from .parse.masking.data_mask import yaml_to_corner_mask_configs


# Libraries
import jax.numpy as jnp
import nifty8.re as jft
from nifty8.logger import logger
import numpy as np

# std
from functools import reduce
from astropy import units as u
from astropy.coordinates import SkyCoord
from typing import Union
from dataclasses import dataclass, field


@dataclass
class FilterData:
    data: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    std: list[np.ndarray] = field(default_factory=list)
    psf_kernel: list[np.ndarray] = field(default_factory=list)
    boresight: list[SkyCoord] = field(default_factory=list)
    transmission: list[float] = field(default_factory=list)
    meta: list[DataMetaInformation] = field(default_factory=list)
    subsample_centers: list[SkyCoord] = field(default_factory=list)
    correction_prior: list[CoordinatesCorrectionConfig] = field(default_factory=list)

    def __len__(self):
        return len(self.data)


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
        filter_data = FilterData()

        for ii, filepath in enumerate(flt):
            logger.info(f"Loading: {fltname} {ii} {filepath}")

            # Loading data, std, and mask.
            jwst_data = JwstData(
                filepath, identifier=f"{fltname}_{ii}", subsample=data_subsample
            )
            energy_name = filter_projector.get_key(jwst_data.pivot_wavelength)
            if ii == 0:
                previous_energy_name = energy_name
            else:
                assert energy_name == previous_energy_name

            data, mask, std, data_subsampled_centers = (
                jwst_data.bounding_data_mask_std_subpixel_by_world_corners(
                    grid.spatial,
                    grid.spatial.world_corners(extension_value=sky_meta.grid_extension),
                    yaml_to_corner_mask_configs(cfg[telescope_key]),
                )
            )
            print(data.shape, data_subsampled_centers.shape)

            psf_kernel = load_psf_kernel(
                jwst_data=jwst_data,
                target_center=grid.spatial.center,
                config_parameters=psf_kernel_configs,
            )

            filter_data.data.append(data)
            filter_data.mask.append(mask)
            filter_data.std.append(std)
            filter_data.subsample_centers.append(data_subsampled_centers)
            filter_data.meta.append(jwst_data.meta)
            filter_data.psf_kernel.append(psf_kernel)
            filter_data.correction_prior.append(
                coordiantes_correction_config.get_name_setting_or_default(
                    jwst_data.filter, ii
                )
            )
            filter_data.boresight.append(jwst_data.get_boresight_world_coords())
            filter_data.transmission.append(jwst_data.transmission)

        for ii in range(len(filter_data)):
            shift_and_rotation_correction = ShiftAndRotationCorrection(
                domain_key=f"{filter_data.meta[ii].identifier}_correction",
                correction_prior=filter_data.correction_prior[ii],
                rotation_center=filter_data.boresight[ii],
            )

            jwst_target_response = build_jwst_response(
                sky_domain={energy_name: filter_projector.target[energy_name]},
                data_subsampled_centers=filter_data.subsample_centers[ii],
                data_meta=filter_data.meta[ii],
                sky_wcs=grid.spatial,
                sky_meta=sky_meta,
                rotation_and_shift_algorithm=rotation_and_shift_algorithm,
                shift_and_rotation_correction=shift_and_rotation_correction,
                psf_kernel=filter_data.psf_kernel[ii],
                transmission=filter_data.transmission[ii],
                zero_flux_prior_config=zero_flux_prior_configs.get_name_setting_or_default(
                    fltname
                ),
                data_mask=filter_data.mask[ii],
            )
            target_plotting[filter_data.meta[ii].identifier] = (
                ResidualPlottingInformation(
                    data=filter_data.data[ii],
                    std=filter_data.std[ii],
                    mask=filter_data.mask[ii],
                    model=jwst_target_response,
                )
            )

            likelihood = build_gaussian_likelihood(
                jnp.array(filter_data.data[ii][filter_data.mask[ii]], dtype=float),
                jnp.array(filter_data.std[ii][filter_data.mask[ii]], dtype=float),
            )
            likelihood = likelihood.amend(
                jwst_target_response, domain=jft.Vector(jwst_target_response.domain)
            )
            likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x + y, likelihoods)
    return likelihood, filter_projector, target_plotting
