from ...likelihood import build_gaussian_likelihood
from ...grid import Grid
from .jwst_response import (
    build_jwst_response,
    TargetResponseInput,
)
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .plotting.residuals import ResidualPlottingInformation
from .plotting.alignment import FilterAlignmentPlottingInformation
from .alignment.filter_alignment import FilterAlignment
from .alignment.star_model import build_star_in_data
from .zero_flux_model import build_zero_flux_model

from .likelihood.target_likelihood import (
    TargetLikelihoodSideEffects,
    build_target_likelihood,
)
from .likelihood.likelihood import build_likelihood

# Parseing
from .parse.jwst_response import SkyMetaInformation
from .parse.parsing_step import ConfigParserJwst
from .config_handler import get_grid_extension_from_config

from .data.loader.data_loader import (
    load_data,
    DataLoader,
    DataLoaderTarget,
    DataLoaderStarAlignment,
)
from .data.preloader.preloader import (
    preload_data,
    Preloader,
    PreloaderSideEffects,
)


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
    sky_meta = SkyMetaInformation(
        grid_extension=get_grid_extension_from_config(cfg[telescope_key], grid),
        unit=sky_unit,
        dvol=grid.spatial.dvol,
    )
    cfg_parser = ConfigParserJwst.from_yaml_dict(
        cfg, telescope_key=telescope_key, files_key=files_key
    )

    target_plotting = ResidualPlottingInformation(
        y_offset=min(filter_projector.keys_and_index.values())
    )
    alignment_plotting = []
    target_filter_likelihoods = []
    stars_alignment_likelihoods = []

    for filter, filepaths in cfg_parser.data_loader.paths.items():
        filter_alignment = FilterAlignment(filter_name=filter)
        filter_alignment.load_correction_prior(
            cfg[telescope_key]["rotation_and_shift"]["correction_priors"],
            number_of_observations=len(filepaths),
        )

        filter_meta, target_bounds, star_tables = preload_data(
            filepaths=filepaths,
            preloader=Preloader(
                grid_corners=grid.spatial.world_corners(
                    extension_value=sky_meta.grid_extension
                ),
                star_alignment_config=cfg_parser.star_alignment_config,
            ),
            side_effects=PreloaderSideEffects(filter_alignment=filter_alignment),
            loading_mode_config=cfg_parser.data_loader.loading_mode_config,
        )

        target_data, stars_data = load_data(
            filepaths=filepaths,
            data_loader=DataLoader(
                target=DataLoaderTarget(
                    grid=grid,
                    data_bounds=target_bounds,
                    subsample=cfg_parser.subsample_target,
                ),
                psf_kernel_configs=cfg_parser.psf_kernel_configs,
                star_alignment=DataLoaderStarAlignment.from_optional(
                    config=cfg_parser.star_alignment_config,
                    tables=star_tables,
                ),
                extra_masks=cfg_parser.extra_masks,
            ),
            loading_mode_config=cfg_parser.data_loader.loading_mode_config,
        )

        shift_and_rotation_correction = ShiftAndRotationCorrection(
            domain_key=filter,
            correction_prior=filter_alignment.correction_prior,
            rotation_center=SkyCoord(filter_alignment.boresight),
        )

        likelihood_target = build_target_likelihood(
            response=TargetResponseInput(
                filter=filter,
                grid=grid,
                filter_projector=filter_projector,
                target_data=target_data,
                filter_meta=filter_meta,
                sky_meta=sky_meta,
                rotation_and_shift_algorithm=cfg_parser.rotation_and_shift_algorithm,
                zero_flux_prior_configs=cfg_parser.zero_flux_prior_configs,
                shift_and_rotation_correction=shift_and_rotation_correction,
            ),
            side_effect=TargetLikelihoodSideEffects(plotting=target_plotting),
        )
        target_filter_likelihoods.append(likelihood_target)

        if cfg_parser.star_alignment_config is not None:
            filter_alignment_plotting = dict(
                psf_convolved=FilterAlignmentPlottingInformation(filter),
                psf=FilterAlignmentPlottingInformation(filter),
            )
            filter_alignment_likelihoods = dict(psf_convolved=[], psf=[])

            # psf_shape = np.array(stars_data[stars[0].id].psf).shape
            # for star in stars:
            #     psf_shape_ii = np.array(stars_data[star.id].psf).shape
            #     assert psf_shape[1:] == psf_shape_ii[1:]
            # psf_model = build_psf_modification_model_strategy(
            #     f"{filter_and_filepaths.filter}", psf_shape, strategy="single"
            # )

            for star in star_tables.get_stars():
                psf = stars_data[star.id].psf
                # psf = build_psf_model_strategy(f"{filter_and_filepaths.filter}_{star.id}", psf_shape, strategy='full')
                # psf = LearnablePsf(psf, psf_model)

                import scipy

                psf_convolved = np.array(
                    [
                        scipy.ndimage.gaussian_filter(psf, 2)
                        for psf in stars_data[star.id].psf
                    ]
                )

                star_data = stars_data[star.id]
                data, mask, std = star_data.data, star_data.mask, star_data.std

                for p, name in zip([psf, psf_convolved], ["psf", "psf_convolved"]):
                    jwst_star_response = build_jwst_response(
                        sky_in_subsampled_data=build_star_in_data(
                            filter_key=filter,
                            filter_meta=filter_meta,
                            star_id=star.id,
                            star_light_prior=cfg_parser.star_alignment_config.star_light_prior,
                            star_data=stars_data[star.id],
                            shift_and_rotation_correction=shift_and_rotation_correction,
                        ),
                        data_meta=filter_meta,
                        data_subsample=stars_data[star.id].subsample,
                        sky_meta=sky_meta,
                        psf=p,
                        zero_flux_model=build_zero_flux_model(
                            f"{filter}_{star.id}",
                            cfg_parser.zero_flux_prior_configs.get_name_setting_or_default(
                                filter
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

                    filter_alignment_likelihoods[name].append(likelihood_star)

                    if name == "psf":
                        filter_alignment_plotting[name].append_information(
                            star_id=star.id,
                            data=np.array(data),
                            mask=np.array(mask),
                            std=np.array(std),
                            model=jwst_star_response,
                        )

            alignment_plotting.append(filter_alignment_plotting["psf"])

            filter_alignment_likelihood_psf = reduce(
                lambda x, y: x + y, filter_alignment_likelihoods["psf"]
            )
            filter_alignment_likelihood_psf_convolved = reduce(
                lambda x, y: x + y, filter_alignment_likelihoods["psf_convolved"]
            )

            stars_alignment_likelihoods.append(
                filter_alignment_likelihood_psf  # +
                # filter_alignment_likelihood_psf_convolved
            )

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
