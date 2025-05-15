from ...likelihood import build_gaussian_likelihood
from ...grid import Grid
from .jwst_response import build_jwst_response, build_sky_to_subsampled_data
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .plotting.residuals import ResidualPlottingInformation
from .plotting.alignment import FilterAlignmentPlottingInformation
from .config_handler import get_grid_extension_from_config
from .alignment.filter_alignment import FilterAlignment
from .alignment.star_alignment import StarAlignment
from .alignment.star_model import build_star_in_data
from .zero_flux_model import build_zero_flux_model

from .data.loading.data_loading import data_loading
from .data.preloading.preloading import data_preloading

# Parsing
from .parse.zero_flux_model import ZeroFluxPriorConfigs
from .parse.rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
)
from .parse.jwst_response import SkyMetaInformation
from .parse.alignment.star_alignment import StarAlignmentMeta
from .parse.data.data_loading import DataFilePaths
from .parse.jwst_psf import JwstPsfKernelConfig
from .parse.masking.data_mask import ExtraMasks

from .psf.psf_learning import build_psf_modification_model_strategy, LearnablePsf

# Libraries
import jax.numpy as jnp
import nifty8.re as jft
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
    zero_flux_prior_configs = ZeroFluxPriorConfigs.from_yaml_dict(
        cfg[telescope_key].get("zero_flux")
    )
    rotation_and_shift_algorithm = rotation_and_shift_algorithm_config_factory(
        cfg[telescope_key]["rotation_and_shift"]
    )
    sky_meta = SkyMetaInformation(
        grid_extension=get_grid_extension_from_config(cfg[telescope_key], grid),
        unit=sky_unit,
        dvol=grid.spatial.dvol,
    )

    psf_kernel_configs = JwstPsfKernelConfig.from_yaml_dict(
        cfg[telescope_key].get("psf")
    )
    extra_masks = ExtraMasks.from_yaml_dict(cfg[telescope_key])

    gaia_alignment_meta_data = StarAlignmentMeta.from_yaml_dict(
        cfg[telescope_key].get("gaia_alignment")
    )

    data_paths = DataFilePaths.from_yaml_dict(cfg[files_key])

    target_plotting = ResidualPlottingInformation()
    alignment_plotting = []
    target_filter_likelihoods = []
    stars_alignment_likelihoods = []

    for filter_and_filepaths in data_paths.filters:
        filter_alignment = FilterAlignment(filter_name=filter_and_filepaths.filter)
        filter_alignment.load_star_alignment(gaia_alignment_meta_data)
        filter_alignment.load_correction_prior(
            cfg[telescope_key]["rotation_and_shift"]["correction_priors"],
            number_of_observations=len(filter_and_filepaths.filepaths),
        )

        color, target_bounds, filter_alignment = data_preloading(
            filter_and_filepaths, grid, sky_meta, filter_alignment
        )

        target_data, stars, stars_data = data_loading(
            telescope_cfg=cfg[telescope_key],
            filter_and_filepaths=filter_and_filepaths,
            target_grid=grid,
            filter_alignment=filter_alignment,
            target_bounds=target_bounds,
            psf_kernel_configs=psf_kernel_configs,
            extra_masks=extra_masks,
        )

        shift_and_rotation_correction = ShiftAndRotationCorrection(
            domain_key=filter_and_filepaths.filter,
            correction_prior=filter_alignment.correction_prior,
            rotation_center=SkyCoord(filter_alignment.boresight),
        )

        energy_name = filter_projector.get_key(color)
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
                f"{filter_and_filepaths.filter}_target",
                zero_flux_prior_configs.get_name_setting_or_default(
                    filter_and_filepaths.filter
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
            filter=filter_and_filepaths.filter,
            data=np.array(target_data.data),
            std=np.array(target_data.std),
            mask=np.array(target_data.mask),
            model=jwst_target_response,
        )
        target_filter_likelihoods.append(likelihood_target)

        star_alignment: StarAlignment | None = filter_alignment.star_alignment
        if star_alignment:
            filter_alignment_plotting = FilterAlignmentPlottingInformation(
                filter_and_filepaths.filter
            )
            filter_alignment_likelihoods = []

            # psf_shape = np.array(stars_data[stars[0].id].psf).shape
            # for star in stars:
            #     psf_shape_ii = np.array(stars_data[star.id].psf).shape
            #     assert psf_shape[1:] == psf_shape_ii[1:]
            # psf_model = build_psf_modification_model_strategy(
            #     f"{filter_and_filepaths.filter}", psf_shape, strategy="single"
            # )

            for star in stars:
                psf = np.array(stars_data[star.id].psf)
                # psf = build_psf_model_strategy(f"{filter_and_filepaths.filter}_{star.id}", psf_shape, strategy='full')
                # psf = LearnablePsf(psf, psf_model)

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
                        filter_key=filter_and_filepaths.filter,
                        star_id=star.id,
                        star_light_prior=star_alignment.alignment_meta.star_light_prior,
                        star_data=stars_data[star.id],
                        shift_and_rotation_correction=shift_and_rotation_correction,
                    ),
                    data_meta=stars_data[star.id].meta,
                    data_subsample=stars_data[star.id].subsample,
                    sky_meta=sky_meta,
                    psf=psf,
                    zero_flux_model=build_zero_flux_model(
                        f"{filter_and_filepaths.filter}_{star.id}",
                        zero_flux_prior_configs.get_name_setting_or_default(
                            filter_and_filepaths.filter
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

    if stars_alignment_likelihoods != []:
        likelihood_alignment = reduce(lambda x, y: x + y, stars_alignment_likelihoods)
    else:
        likelihood_alignment = None

    return (
        likelihood_target,
        filter_projector,
        target_plotting,
        likelihood_alignment,
        alignment_plotting if alignment_plotting != [] else None,
    )
