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
from ..parse.alignment.star_alignment import StarAlignmentConfig
from ..alignment.star_model import build_star_in_data
from ..data.loader.stars_loader import StarsData
from ..zero_flux_model import build_zero_flux_model
from ..parse.zero_flux_model import ZeroFluxPriorConfigs
from ..rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from ..data.jwst_data import DataMetaInformation
from ..parse.jwst_response import SkyMetaInformation

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
    star_light_prior: StarAlignmentConfig
    shift_and_rotation_correction: ShiftAndRotationCorrection | None
    zero_flux_prior_configs: ZeroFluxPriorConfigs


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

    # psf_shape = np.array(stars_data[stars[0].id].psf).shape
    # for star in stars:
    #     psf_shape_ii = np.array(stars_data[star.id].psf).shape
    #     assert psf_shape[1:] == psf_shape_ii[1:]
    # psf_model = build_psf_modification_model_strategy(
    #     f"{filter_and_filepaths.response.filter_name}", psf_shape, strategy="single"
    # )

    for star in response.star_tables.get_stars():
        # Check if star.id in
        if star.id not in response.stars_data.keys():
            jft.logger.info(
                f"Warning: {star.id} is not in filter {response.filter_name}"
            )
            continue

        psf = response.stars_data[star.id].psf
        # psf = build_psf_model_strategy(f"{filter_and_filepaths.response.filter_name}_{star.id}", psf_shape, strategy='full')
        # psf = LearnablePsf(psf, psf_model)

        import scipy

        psf_convolved = np.array(
            [
                scipy.ndimage.gaussian_filter(psf, 2)
                for psf in response.stars_data[star.id].psf
            ]
        )

        star_data = response.stars_data[star.id]
        data, mask, std = star_data.data, star_data.mask, star_data.std

        for p, name in zip([psf, psf_convolved], ["psf", "psf_convolved"]):
            jwst_star_response: JwstResponse = build_jwst_response(
                sky_in_subsampled_data=build_star_in_data(
                    filter_key=response.filter_name,
                    filter_meta=response.filter_meta,
                    star_id=star.id,
                    star_light_prior=response.star_light_prior,
                    star_data=response.stars_data[star.id],
                    shift_and_rotation_correction=response.shift_and_rotation_correction,
                ),
                data_meta=response.filter_meta,
                data_subsample=response.stars_data[star.id].subsample,
                sky_meta=response.sky_meta,
                psf=p,
                zero_flux_model=build_zero_flux_model(
                    f"{response.filter_name}_{star.id}",
                    response.zero_flux_prior_configs.get_name_setting_or_default(
                        response.filter_name
                    ),
                    shape=(len(response.stars_data[star.id].data), 1, 1),
                ),
                data_mask=np.array(response.stars_data[star.id].mask),
            )

            likelihood_star = GaussianLikelihoodBuilder(
                response=jwst_star_response,
                data=response.stars_data[star.id].data,
                std=response.stars_data[star.id].std,
                mask=response.stars_data[star.id].mask,
            ).build()

            filter_alignment_likelihoods[name].append(likelihood_star)
            filter_alignment_plotting[name].append_information(
                star_id=star.id,
                data=np.array(data),
                mask=np.array(mask),
                std=np.array(std),
                model=jwst_star_response,
                subsample=response.stars_data[star.id].subsample,
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
