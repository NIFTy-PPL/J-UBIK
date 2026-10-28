from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import nifty.re as jft
from ..jwst_response import (
    JwstResponse,
    build_jwst_response,
)
from ..plotting.alignment import (
    FilterAlignmentPlottingInformation,
    MultiFilterAlignmentPlottingInformation,
)
from .likelihood import GaussianLikelihoodBuilder
from ..alignment.star_alignment import StarTables
from ..alignment.star_model import build_star_in_data, build_star_position_offset
from ..data.loader.stars_loader import StarsData
from ..zero_flux_model import build_zero_flux_model
from ..parse.zero_flux_model import ZeroFluxPriorConfigs
from ..rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from ..data.jwst_data import DataMetaInformation
from ..parse.jwst_response import SkyMetaInformation
from ....parse.models.distributions import ProbabilityConfig
from ....wcs.wcs_astropy import WcsAstropy

# --------------------------------------------------------------------------------------
# Alignment Likelihood API
# --------------------------------------------------------------------------------------


@dataclass
class AlignmentLikelihoodSideEffects:
    plotting: MultiFilterAlignmentPlottingInformation


@dataclass
class StarAlignmentResponseInput:
    filter_name: str
    filter_meta: DataMetaInformation
    sky_meta: SkyMetaInformation
    star_tables: StarTables
    stars_data: StarsData
    star_light_prior: ProbabilityConfig
    shift_and_rotation_correction: ShiftAndRotationCorrection | None
    zero_flux_prior_configs: ZeroFluxPriorConfigs
    fit_star_positions: bool
    grid_wcs: WcsAstropy


@dataclass
class AlignmentLikelihoodProducts:
    likelihood: jft.Likelihood
    likelihood_convolved: jft.Likelihood | None


@dataclass
class MultiFilterAlignmentLikelihoods:
    likelihoods: list[AlignmentLikelihoodProducts] = field(default_factory=list)

    @property
    def likelihood(self):
        return reduce(lambda x, y: x + y, (ll.likelihood for ll in self.likelihoods))

    @property
    def likelihood_convolved(self):
        return reduce(
            lambda x, y: x + y, (ll.likelihood_convolved for ll in self.likelihoods)
        )


def build_star_alignment_likelihood(
    response: StarAlignmentResponseInput,
    side_effect: AlignmentLikelihoodSideEffects,
) -> AlignmentLikelihoodProducts:
    filter_alignment_plotting = dict(
        psf_convolved=FilterAlignmentPlottingInformation(response.filter_name),
        psf=FilterAlignmentPlottingInformation(response.filter_name),
    )
    filter_alignment_likelihoods = dict(psf_convolved=[], psf=[])

    zero_flux_prior_config = (
        None
        if response.zero_flux_prior_configs is None
        else response.zero_flux_prior_configs.get_name_setting_or_default(
            response.filter_name
        )
    )

    for star in response.star_tables.get_stars():
        if star.id not in response.stars_data.keys():
            jft.logger.info(
                f"Warning: {star.id} is not in filter {response.filter_name}"
            )
            continue

        star_data = response.stars_data[star.id]
        data, mask, std = star_data.data, star_data.mask, star_data.std
        psf = star_data.psf

        import scipy

        psf_convolved = np.array([scipy.ndimage.gaussian_filter(p, 2) for p in psf])

        position = None
        if response.fit_star_positions:
            covariance = star.astrometric_covariance
            if covariance is None or np.isnan(covariance).any():
                raise ValueError(
                    f"Gaia star {star.id} has no astrometric covariance: an error or "
                    "correlation of ra, dec, pmra or pmdec is missing, masked or NaN "
                    "(a star table cached without error columns needs a fresh "
                    "query). Exclude it with `exclude_source_id` or turn off "
                    "`fit_star_positions`."
                )
            times = response.star_tables.observation_times
            position = build_star_position_offset(
                star, response.grid_wcs, [times[i] for i in star_data.observation_ids]
            )

        for p, name in zip([psf, psf_convolved], ["psf", "psf_convolved"]):
            jwst_star_response: JwstResponse = build_jwst_response(
                sky_in_subsampled_data=build_star_in_data(
                    filter_key=response.filter_name,
                    star_id=star.id,
                    star_light_prior=response.star_light_prior,
                    star_data=star_data,
                    shift_and_rotation_correction=response.shift_and_rotation_correction,
                    position=position,
                ),
                data_meta=response.filter_meta,
                data_subsample=star_data.subsample,
                sky_meta=response.sky_meta,
                psf=p,
                zero_flux_model=build_zero_flux_model(
                    f"{response.filter_name}_{star.id}",
                    zero_flux_prior_config,
                    shape=(len(star_data.data), 1, 1),
                ),
                data_mask=np.array(star_data.mask),
            )

            likelihood_star = GaussianLikelihoodBuilder(
                response=jwst_star_response,
                data=star_data.data,
                std=star_data.std,
                mask=star_data.mask,
            ).build()

            filter_alignment_likelihoods[name].append(likelihood_star)
            filter_alignment_plotting[name].append_information(
                star_id=star.id,
                data=np.array(data),
                mask=np.array(mask),
                std=np.array(std),
                model=jwst_star_response,
                subsample=star_data.subsample,
            )

    if not filter_alignment_likelihoods["psf"]:
        raise ValueError(
            f"Star alignment for filter {response.filter_name!r} has no usable "
            f"star: {len(response.star_tables.get_stars())} catalog stars, "
            f"{len(response.stars_data)} with a valid cutout. Either the Gaia "
            "query returned nothing for this field or every candidate was "
            "rejected during loading (position off the detector or an all-NaN "
            "cutout). Remove the `gaia_alignment` block or check the field."
        )

    side_effect.plotting.psf.append(filter_alignment_plotting["psf"])
    side_effect.plotting.convolved.append(filter_alignment_plotting["psf_convolved"])

    filter_alignment_likelihood_psf = reduce(
        lambda x, y: x + y, filter_alignment_likelihoods["psf"]
    )
    filter_alignment_likelihood_psf_convolved = reduce(
        lambda x, y: x + y, filter_alignment_likelihoods["psf_convolved"]
    )

    return AlignmentLikelihoodProducts(
        likelihood=filter_alignment_likelihood_psf,
        likelihood_convolved=filter_alignment_likelihood_psf_convolved,
    )
