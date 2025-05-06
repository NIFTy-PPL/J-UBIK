from ...likelihood import build_gaussian_likelihood
from ...grid import Grid
from ...wcs.wcs_jwst_data import WcsJwstData
from .jwst_response import build_jwst_response, build_jwst_response_stars
from .data.jwst_data import JwstData
from .data.data_loading import DataBoundsPreloading
from .jwst_psf import load_psf_kernel
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
)
from .plotting.residuals import ResidualPlottingInformation
from .config_handler import (
    get_grid_extension_from_config,
)
from ..gaia.star_finder import load_gaia_stars_in_fov, join_tables
from .plotting.plotting_sky import plot_sky_coords, plot_jwst_panels
from .alignment.star_alignment import FilterAlignment
from .alignment.star_model import build_star_in_data
from .zero_flux_model import build_zero_flux_model

# Parsing
from .parse.jwst_psf import yaml_to_psf_kernel_config
from .parse.zero_flux_model import yaml_to_zero_flux_prior_config
from .parse.rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
)
from .parse.jwst_response import SkyMetaInformation
from .parse.masking.data_mask import yaml_to_corner_mask_configs
from .parse.alignment.star_alignment import FilterAlignmentMeta
from .parse.data.data_loading import DataFilePaths
from .parse.jwst_likelihoods import TargetData, StarData


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
    rotation_and_shift_algorithm = rotation_and_shift_algorithm_config_factory(
        cfg[telescope_key]["rotation_and_shift"]
    )
    data_subsample = cfg[telescope_key]["rotation_and_shift"]["subsample"]
    sky_meta = SkyMetaInformation(
        grid_extension=get_grid_extension_from_config(cfg[telescope_key], grid),
        unit=sky_unit,
    )

    gaia_alignment_meta_data = FilterAlignmentMeta.from_yaml_dict(
        cfg[telescope_key].get("gaia_alignment", {})
    )

    data_paths = DataFilePaths.from_yaml_dict(cfg[files_key])

    target_plotting = ResidualPlottingInformation()
    likelihoods = []
    for filter_and_files in data_paths.filters:
        target_preloading = DataBoundsPreloading()

        filter_alignment = FilterAlignment(
            filter_name=filter_and_files.name, alignment_meta=gaia_alignment_meta_data
        )
        filter_alignment.load_correction_prior(
            cfg[telescope_key]["rotation_and_shift"]["correction_priors"],
            number_of_observations=len(filter_and_files.filepaths),
        )

        # NOTE : Preloading Data
        for ii, filepath in enumerate(filter_and_files.filepaths):
            print(ii, filepath)
            jwst_data = JwstData(filepath, subsample=data_subsample)
            energy_name = filter_projector.get_key(jwst_data.pivot_wavelength)
            if ii == 0:
                previous_energy_name = energy_name
            else:
                assert energy_name == previous_energy_name

            filter_alignment.star_tables.append(
                load_gaia_stars_in_fov(
                    jwst_data.wcs.world_corners(),
                    filter_alignment.alignment_meta.library_path,
                )
            )

            target_preloading.append_shapes_and_bounds(
                jwst_data,
                sky_corners=grid.spatial.world_corners(
                    extension_value=sky_meta.grid_extension
                ),
            )

        target_preloading = target_preloading.align_shapes()

        target_data = TargetData()

        stars = filter_alignment.get_stars()
        stars_data = {star.id: StarData() for star in stars}

        for observation_id, filepath in enumerate(filter_and_files.filepaths):
            logger.info(f"Loading: {filter_and_files.name} {observation_id} {filepath}")

            # Loading data, std, and mask.
            jwst_data = JwstData(filepath, subsample=data_subsample)
            filter_alignment.boresight.append(jwst_data.get_boresight_world_coords())

            data, mask, std, data_subsampled_centers = (
                jwst_data.bounding_data_mask_std_subpixel_by_bounding_indices(
                    grid.spatial,
                    target_preloading.bounding_indices[observation_id],
                    yaml_to_corner_mask_configs(cfg[telescope_key]),
                )
            )
            psf = load_psf_kernel(
                jwst_data=jwst_data,
                target_center=grid.spatial.center,
                config_parameters=psf_kernel_configs,
            )
            target_data.append_observation(
                meta=jwst_data.meta,
                data=data,
                mask=mask,
                std=std,
                subsample_centers=data_subsampled_centers,
                psf=psf,
            )

            for ii, star in enumerate(stars):
                # FIXME : DELETE THIS
                print("Warning: delete me")
                if ii > 1:
                    continue

                star_data = stars_data[star.id]

                if jwst_data.position_outside_data(star.position):
                    continue

                fov_pixel = (
                    filter_alignment.alignment_meta.fov.to(u.arcsec)
                    / jwst_data.meta.pixel_distance.to(u.arcsec)
                ).value
                fov_pixel = (int(np.round(fov_pixel)),) * 2

                data, mask, std, data_subsampled_centers = (
                    jwst_data.bounding_data_mask_std_subpixel_by_bounding_indices(
                        None,
                        star.bounding_indices(jwst_data, fov_pixel),
                        yaml_to_corner_mask_configs(cfg[telescope_key]),
                    )
                )
                psf = load_psf_kernel(
                    jwst_data=jwst_data,
                    target_center=star.position,
                    config_parameters=psf_kernel_configs,
                )
                star_in_subsampled_pixels = star.subpixel_position_in_world_coordinates(
                    data_subsampled_centers
                )

                star_data.append_observation(
                    meta=jwst_data.meta,
                    data=data,
                    mask=mask,
                    std=std,
                    sky_array=np.zeros(data_subsampled_centers.shape),
                    psf=psf,
                    star_in_subsampled_pixles=star_in_subsampled_pixels,
                    observation_id=observation_id,
                )

        shift_and_rotation_correction = ShiftAndRotationCorrection(
            domain_key=filter_and_files.name,
            correction_prior=filter_alignment.correction_prior,
            rotation_center=SkyCoord(filter_alignment.boresight),
        )

        zero_flux_model = build_zero_flux_model(
            filter_and_files.name,
            zero_flux_prior_configs.get_name_setting_or_default(filter_and_files.name),
            shape=(len(target_data), 1, 1),
        )
        for star in stars:
            star_in_subsampled_data_field = build_star_in_data(
                filter_key=filter_and_files.name,
                star=star,
                star_light_prior=filter_alignment.alignment_meta.star_light_prior,
                shift_and_rotation_correction=shift_and_rotation_correction,
            )
            exit()

            jwst_stars_response = build_jwst_response_stars(
                star=star,
                star_data=stars_data[star.id],
                data_meta=target_data.meta,
                sky_wcs=grid.spatial,
                sky_meta=sky_meta,
                shift_and_rotation_correction=shift_and_rotation_correction,
                psf=np.array(target_data.psf),
                zero_flux_model=zero_flux_model,
                data_mask=np.array(target_data.mask),
            )
            exit()

        jwst_target_response = build_jwst_response(
            sky_domain={energy_name: filter_projector.target[energy_name]},
            data_subsampled_centers=target_data.subsample_centers,
            data_meta=target_data.meta,
            sky_wcs=grid.spatial,
            sky_meta=sky_meta,
            rotation_and_shift_algorithm=rotation_and_shift_algorithm,
            shift_and_rotation_correction=shift_and_rotation_correction,
            psf=np.array(target_data.psf),
            zero_flux_model=zero_flux_model,
            data_mask=np.array(target_data.mask),
        )

        likelihood = build_gaussian_likelihood(
            jnp.array(
                np.array(target_data.data)[np.array(target_data.mask)], dtype=float
            ),
            jnp.array(
                np.array(target_data.std)[np.array(target_data.mask)], dtype=float
            ),
        )
        likelihood = likelihood.amend(
            jwst_target_response, domain=jft.Vector(jwst_target_response.domain)
        )

        target_plotting.append_information(
            filter=filter_and_files.name,
            data=np.array(target_data.data),
            std=np.array(target_data.std),
            mask=np.array(target_data.mask),
            model=jwst_target_response,
        )
        likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x + y, likelihoods)

    return likelihood, filter_projector, target_plotting
