from ...likelihood import build_gaussian_likelihood
from ...grid import Grid
from ..gaia.star_finder import load_gaia_stars_in_fov
from .jwst_response import build_jwst_response, build_sky_to_subsampled_data
from .data.jwst_data import JwstData
from .data.data_loading import DataBoundsPreloading
from .jwst_psf import load_psf_kernel, build_psf_modification_model_strategy, PsfModel
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .plotting.residuals import ResidualPlottingInformation
from .plotting.alignment import FilterAlignmentPlottingInformation
from .config_handler import get_grid_extension_from_config
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
    files_key: str = "files",
    telescope_key: str = "telescope",
    sky_unit: u.Unit | None = None,
) -> Union[jft.Likelihood, FilterProjector, dict]:
    """Build the jwst likelihood_target according to the config and grid."""

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
    target_subsample = cfg[telescope_key]["rotation_and_shift"]["subsample"]
    alignment_subsample = cfg[telescope_key]["gaia_alignment"]["subsample"]
    sky_meta = SkyMetaInformation(
        grid_extension=get_grid_extension_from_config(cfg[telescope_key], grid),
        unit=sky_unit,
        dvol=grid.spatial.dvol,
    )

    gaia_alignment_meta_data = FilterAlignmentMeta.from_yaml_dict(
        cfg[telescope_key].get("gaia_alignment", {})
    )

    data_paths = DataFilePaths.from_yaml_dict(cfg[files_key])

    target_plotting = ResidualPlottingInformation()
    alignment_plotting = []
    target_filter_likelihoods = []
    stars_alignment_likelihoods = []
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
            jwst_data = JwstData(filepath)
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

            bounding_indices = jwst_data.wcs.bounding_indices_from_world_extrema(
                grid.spatial.world_corners(extension_value=sky_meta.grid_extension)
            )
            target_preloading.append_shapes_and_bounds(
                jwst_data=jwst_data,
                bounding_indices=bounding_indices,
            )

        target_preloading = target_preloading.align_shapes()

        target_data = TargetData()
        stars = filter_alignment.get_stars()
        stars_data = {star.id: StarData() for star in stars}

        for observation_id, filepath in enumerate(filter_and_files.filepaths):
            logger.info(f"Loading: {filter_and_files.name} {observation_id} {filepath}")

            # Loading data, std, and mask.
            jwst_data = JwstData(filepath)
            filter_alignment.boresight.append(jwst_data.get_boresight_world_coords())

            data, mask, std = jwst_data.bounding_data_mask_std_by_bounding_indices(
                target_preloading.bounding_indices[observation_id],
                grid.spatial,
                yaml_to_corner_mask_configs(cfg[telescope_key]),
            )
            target_data.append_observation(
                meta=jwst_data.meta,
                subsample=target_subsample,
                data=data,
                mask=mask,
                std=std,
                subsample_centers=jwst_data.data_subpixel_centers(
                    target_preloading.bounding_indices[observation_id],
                    subsample=target_subsample,
                ),
                psf=load_psf_kernel(
                    jwst_data=jwst_data,
                    subsample=target_subsample,
                    target_center=grid.spatial.center,
                    config_parameters=psf_kernel_configs,
                ),
            )

            for star in filter_alignment.get_stars(observation_id):
                fov_pixel = (
                    filter_alignment.alignment_meta.fov.to(u.arcsec)
                    / jwst_data.meta.pixel_distance.to(u.arcsec)
                ).value
                fov_pixel = np.array((int(np.round(fov_pixel)),) * 2)
                if (fov_pixel % 2).sum() == 0:
                    fov_pixel += 1

                bounding_indices = star.bounding_indices(jwst_data, fov_pixel)
                data, mask, std = jwst_data.bounding_data_mask_std_by_bounding_indices(
                    row_minmax_column_minmax=bounding_indices,
                    additional_masks_corners=yaml_to_corner_mask_configs(
                        cfg[telescope_key]
                    ),
                )

                stars_data[star.id].append_observation(
                    meta=jwst_data.meta,
                    subsample=alignment_subsample,
                    data=data,
                    mask=mask,
                    std=std,
                    psf=load_psf_kernel(
                        jwst_data=jwst_data,
                        subsample=alignment_subsample,
                        target_center=star.position,
                        config_parameters=psf_kernel_configs,
                    ),
                    sky_array=np.zeros([s * alignment_subsample for s in data.shape]),
                    star_in_subsampled_pixles=star.pixel_position_in_subsampled_data(
                        jwst_data.wcs,
                        min_row=bounding_indices[0],
                        min_column=bounding_indices[2],
                        subsample_factor=alignment_subsample,
                    ),
                    observation_id=observation_id,
                )

        shift_and_rotation_correction = ShiftAndRotationCorrection(
            domain_key=filter_and_files.name,
            correction_prior=filter_alignment.correction_prior,
            rotation_center=SkyCoord(filter_alignment.boresight),
        )

        jwst_target_response = build_jwst_response(
            sky_in_subsampled_data=build_sky_to_subsampled_data(
                sky_domain={energy_name: filter_projector.target[energy_name]},
                data_subsampled_centers=target_data.subsample_centers,
                sky_wcs=grid.spatial,
                rotation_and_shift_algorithm=rotation_and_shift_algorithm,
                shift_and_rotation_correction=shift_and_rotation_correction,
            ),
            data_meta=target_data.meta,
            data_subsample=target_data.subsample,
            sky_meta=sky_meta,
            psf=np.array(target_data.psf),
            zero_flux_model=build_zero_flux_model(
                f"{filter_and_files.name}_target",
                zero_flux_prior_configs.get_name_setting_or_default(
                    filter_and_files.name
                ),
                shape=(len(target_data), 1, 1),
            ),
            data_mask=np.array(target_data.mask),
        )

        likelihood_target = build_gaussian_likelihood(
            jnp.array(
                np.array(target_data.data)[np.array(target_data.mask)], dtype=float
            ),
            jnp.array(
                np.array(target_data.std)[np.array(target_data.mask)], dtype=float
            ),
            model=jwst_target_response,
        )

        target_plotting.append_information(
            filter=filter_and_files.name,
            data=np.array(target_data.data),
            std=np.array(target_data.std),
            mask=np.array(target_data.mask),
            model=jwst_target_response,
        )
        target_filter_likelihoods.append(likelihood_target)

        filter_alignment_plotting = FilterAlignmentPlottingInformation(
            filter_and_files.name
        )
        filter_alignment_likelihoods = []

        psf_shape = np.array(stars_data[stars[0].id].psf).shape
        for star in stars:
            psf_shape_ii = np.array(stars_data[star.id].psf).shape
            assert psf_shape[1:] == psf_shape_ii[1:]
        psf_model = build_psf_modification_model_strategy(
            f"{filter_and_files.name}", psf_shape, strategy="single"
        )

        for star in stars:
            psf = np.array(stars_data[star.id].psf)
            # psf = build_psf_model_strategy(f"{filter_and_files.name}_{star.id}", psf_shape, strategy='full')
            # psf = PsfModel(psf, psf_model)

            # import scipy
            #
            # psf = np.array(
            #     [
            #         scipy.ndimage.gaussian_filter(psf, 2)
            #         for psf in stars_data[star.id].psf
            #     ]
            # )

            star_data = stars_data[star.id]
            data, mask, std = star_data.data, star_data.mask, star_data.std

            jwst_star_response = build_jwst_response(
                sky_in_subsampled_data=build_star_in_data(
                    filter_key=filter_and_files.name,
                    star_id=star.id,
                    star_light_prior=filter_alignment.alignment_meta.star_light_prior,
                    star_data=stars_data[star.id],
                    shift_and_rotation_correction=shift_and_rotation_correction,
                ),
                data_meta=stars_data[star.id].meta,
                data_subsample=stars_data[star.id].subsample,
                sky_meta=sky_meta,
                psf=psf,
                zero_flux_model=build_zero_flux_model(
                    f"{filter_and_files.name}_{star.id}",
                    zero_flux_prior_configs.get_name_setting_or_default(
                        filter_and_files.name
                    ),
                    shape=(len(stars_data[star.id].data), 1, 1),
                ),
                data_mask=np.array(stars_data[star.id].mask),
            )

            likelihood_star = build_gaussian_likelihood(
                jnp.array(np.array(data)[np.array(mask)], dtype=float),
                jnp.array(np.array(std)[np.array(mask)], dtype=float),
                model=jwst_star_response,
            )

            filter_alignment_likelihoods.append(likelihood_star)
            filter_alignment_plotting.append_information(
                star_id=star.id,
                data=np.array(data),
                mask=np.array(mask),
                std=np.array(std),
                model=jwst_star_response,
            )
        alignment_plotting.append(filter_alignment_plotting)

        filter_alignment_likelihood = reduce(
            lambda x, y: x + y, filter_alignment_likelihoods
        )

        stars_alignment_likelihoods.append(filter_alignment_likelihood)

    likelihood_target = reduce(lambda x, y: x + y, target_filter_likelihoods)
    likelihood_alignment = reduce(lambda x, y: x + y, stars_alignment_likelihoods)

    return (
        likelihood_target,
        filter_projector,
        target_plotting,
        likelihood_alignment,
        alignment_plotting,
    )
