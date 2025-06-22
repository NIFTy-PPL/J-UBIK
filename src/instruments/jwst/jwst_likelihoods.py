from dataclasses import dataclass


from ...grid import Grid
from .jwst_response import build_target_response, TargetResponseInput
from .filter_projector import FilterProjector, build_filter_projector
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .plotting.residuals import ResidualPlottingInformation
from .plotting.alignment import MultiFilterAlignmentPlottingInformation
from .alignment.filter_alignment import FilterAlignment
from .variable_covariance.inverse_standard_deviation import (
    build_inverse_standard_deviation,
)

from .likelihood.target_likelihood import (
    SingleTargetLikelihood,
    TargetLikelihoodSideEffects,
    build_target_likelihood,
)
from .likelihood.alignment_likelihood import (
    build_star_alignment_likelihood,
    StarAlignmentResponseInput,
    AlignmentLikelihoodSideEffects,
    MultiFilterAlignmentLikelihoods,
)

from .minimization.mask_hot_pixel_data import HotPixelMaskingData

# Parsing
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


@dataclass
class TargetLikelihoodProducts:
    likelihoods: list[SingleTargetLikelihood]
    plotting: ResidualPlottingInformation
    filter_projector: FilterProjector | None = None
    hot_pixel_masking_data: HotPixelMaskingData | None = None

    @property
    def likelihood(self) -> jft.Likelihood | jft.Gaussian:
        likelihoods = (t.likelihood for t in self.likelihoods)
        return reduce(lambda x, y: x + y, likelihoods)


@dataclass
class AlignemntLikelihoodProducts:
    likelihood: MultiFilterAlignmentLikelihoods
    plotting: MultiFilterAlignmentPlottingInformation

    @classmethod
    def from_optional(
        cls,
        likelihood: MultiFilterAlignmentLikelihoods | None,
        plotting: MultiFilterAlignmentPlottingInformation | None,
    ):
        if likelihood is None:
            return None
        return cls(likelihood=likelihood, plotting=plotting)


@dataclass
class JwstLikelihoodProducts:
    target: TargetLikelihoodProducts
    alignment: AlignemntLikelihoodProducts | None


def build_jwst_likelihoods(
    cfg: dict,
    grid: Grid,
    sky_model: jft.Model,
    files_key: str = "files",
    telescope_key: str = "telescope",
    sky_unit: u.Unit | None = None,
) -> JwstLikelihoodProducts:
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
    target_filter_likelihoods = []
    alignment_plotting = (
        MultiFilterAlignmentPlottingInformation()
        if cfg_parser.star_alignment_config is not None
        else None
    )
    alignment_likelihoods = (
        MultiFilterAlignmentLikelihoods()
        if cfg_parser.star_alignment_config is not None
        else None
    )

    hot_pixel_masking_data = HotPixelMaskingData()

    for filter, filepaths in cfg_parser.data_loader.paths.items():
        filter_alignment = FilterAlignment(filter_name=filter)

        preload_results = preload_data(
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

        dataload_results = load_data(
            filepaths=filepaths,
            data_loader=DataLoader(
                target=DataLoaderTarget.from_optional(
                    grid=grid,
                    data_bounds=preload_results.target_bounds,
                    subsample=cfg_parser.subsample_target,
                ),
                psf_kernel_configs=cfg_parser.psf_kernel_configs,
                star_alignment=DataLoaderStarAlignment.from_optional(
                    config=cfg_parser.star_alignment_config,
                    tables=preload_results.star_tables,
                ),
                extra_masks=cfg_parser.extra_masks,
            ),
            loading_mode_config=cfg_parser.data_loader.loading_mode_config,
        )

        hot_pixel_masking_data.append_information(
            filter=filter, nan_mask=dataload_results.target_data.nan_mask
        )

        # Constructing the Likelihood
        filter_alignment.load_correction_prior(
            cfg[telescope_key]["rotation_and_shift"]["correction_priors"],
            number_of_observations=len(filepaths),
        )
        shift_and_rotation_correction = ShiftAndRotationCorrection(
            domain_key=filter,
            correction_prior=filter_alignment.correction_prior,
            rotation_center=SkyCoord(filter_alignment.boresight),
        )

        likelihood_target: SingleTargetLikelihood = build_target_likelihood(
            response=build_target_response(
                input_config=TargetResponseInput(
                    filter_name=filter,
                    grid=grid,
                    filter_projector=filter_projector,
                    target_data=dataload_results.target_data,
                    filter_meta=preload_results.filter_meta,
                    sky_meta=sky_meta,
                    rotation_and_shift_algorithm=cfg_parser.rotation_and_shift_algorithm,
                    zero_flux_prior_configs=cfg_parser.zero_flux_prior_configs,
                    shift_and_rotation_correction=shift_and_rotation_correction,
                )
            ),
            target_data=dataload_results.target_data,
            filter_name=filter,
            inverse_std_builder=build_inverse_standard_deviation(
                config=cfg_parser.variable_covariance_config,
                filter_name=filter,
                target_data=dataload_results.target_data,
            ),
            side_effect=TargetLikelihoodSideEffects(plotting=target_plotting),
        )
        target_filter_likelihoods.append(likelihood_target)

        if cfg_parser.star_alignment_config is not None:
            filter_alignment_likelihood = build_star_alignment_likelihood(
                response=StarAlignmentResponseInput(
                    filter_name=filter,
                    filter_meta=preload_results.filter_meta,
                    sky_meta=sky_meta,
                    star_tables=preload_results.star_tables,
                    stars_data=dataload_results.stars_data,
                    star_light_prior=cfg_parser.star_alignment_config.star_light_prior,
                    shift_and_rotation_correction=shift_and_rotation_correction,
                    zero_flux_prior_configs=cfg_parser.zero_flux_prior_configs,
                ),
                side_effect=AlignmentLikelihoodSideEffects(plotting=alignment_plotting),
            )
            alignment_likelihoods.likelihoods.append(filter_alignment_likelihood)

    return JwstLikelihoodProducts(
        target=TargetLikelihoodProducts(
            likelihoods=target_filter_likelihoods,
            plotting=target_plotting,
            filter_projector=filter_projector,
            hot_pixel_masking_data=hot_pixel_masking_data,
        ),
        alignment=AlignemntLikelihoodProducts.from_optional(
            likelihood=alignment_likelihoods,
            plotting=alignment_plotting,
        ),
    )
