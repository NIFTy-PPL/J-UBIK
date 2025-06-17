from typing import Optional
from dataclasses import dataclass, asdict

import numpy as np
import nifty8.re as jft

from ....likelihood import connect_likelihood_to_model
from ..likelihood.likelihood import (
    LikelihoodData,
    GaussianLikelihoodInput,
)
from ..likelihood.target_likelihood import SingleTargetLikelihood

from ....minimization_parser import MinimizationParser
from ....minimization.minimization_from_samples import (
    KLSettings,
    minimization_from_initial_samples,
)
from ..jwst_likelihoods import TargetLikelihoodProducts
from ..plotting.residuals import ResidualPlottingInformation


@dataclass
class HotPixelMasking:
    mask_at_step: int
    sigma: float
    sky_with_filter: jft.Model

    def adjust_kl_settings(
        self, kl_settings: KLSettings, before_masking: bool = False
    ) -> KLSettings:
        if before_masking:
            start = 0
            end = self.mask_at_step
            resume = kl_settings.resume
        else:
            start = self.mask_at_step
            end = kl_settings.n_total_iterations
            resume = True

        klset = asdict(kl_settings)
        klset["minimization"] = MinimizationParserUpdate.minimization_at_iteration(
            kl_settings.minimization, start
        )
        klset["n_total_iterations"] = end
        klset["resume"] = resume

        return KLSettings(**klset)

    @classmethod
    def from_yaml_dict(cls, raw: dict, sky_with_filter: jft.Model) -> "HotPixelMasking":
        return cls(
            mask_at_step=raw["hot_pixel"]["mask_at_step"],
            sigma=raw["hot_pixel"]["sigma"],
            sky_with_filter=sky_with_filter,
        )


def _under_the_hood(dm, d, m, s, R, mask_hot_pixel):
    res = (d[m] - dm) / s[m]
    extra_m = np.abs(res) < mask_hot_pixel.sigma
    m[m] = extra_m

    return jft.Model(lambda x: R(x)[extra_m], domain=R.domain)


def masking_hot_pixels(
    likelihood: TargetLikelihoodProducts,
    plotting: ResidualPlottingInformation,
    samples: jft.Samples,
    mask_hot_pixel: HotPixelMasking,
) -> TargetLikelihoodProducts:
    def response(si, R):
        return R(mask_hot_pixel.sky_with_filter(si) | si.tree)

    target_plotting = ResidualPlottingInformation(y_offset=plotting.y_offset)

    new_likelihoods = []
    for ll in likelihood.likelihoods:
        d = ll.builder.data.data
        m = ll.builder.data.mask
        s = ll.builder.data.std
        R = ll.builder.response

        dm = jft.mean([response(si, R) for si in samples])
        response_new = _under_the_hood(dm, d, m, s, R, mask_hot_pixel)

        builder = GaussianLikelihoodInput(
            response=response_new,
            data=LikelihoodData(data=d, std=s, mask=m),
        )

        target_plotting.append_information(
            filter=ll.filter,
            data=d,
            mask=m,
            std=s,
            model=response_new,
        )

        new_likelihoods.append(
            SingleTargetLikelihood(filter=ll.filter, builder=builder)
        )

    return TargetLikelihoodProducts(
        likelihoods=new_likelihoods,
        plotting=target_plotting,
        filter_projector=likelihood.filter_projector,
    )


@dataclass
class MinimizationParserUpdate:
    n_samples: callable
    sample_mode: callable
    draw_linear_kwargs: callable
    nonlinearly_update_kwargs: callable
    kl_kwargs: callable

    @classmethod
    def minimization_at_iteration(cls, parser: MinimizationParser, iteration: int):
        return cls(
            n_samples=lambda ii: parser.n_samples(ii + iteration),
            sample_mode=lambda ii: parser.sample_mode(ii + iteration),
            draw_linear_kwargs=lambda ii: parser.draw_linear_kwargs(ii + iteration),
            nonlinearly_update_kwargs=lambda ii: parser.nonlinearly_update_kwargs(
                ii + iteration
            ),
            kl_kwargs=lambda ii: parser.kl_kwargs(ii + iteration),
        )


def minimize_with_hot_pixel_masking(
    likelihood: TargetLikelihoodProducts,
    kl_settings: KLSettings,
    masking: HotPixelMasking,
    starting_samples: Optional[jft.Samples] = None,
    not_take_starting_pos_keys: tuple[str] = (),
):
    samples, state = minimization_from_initial_samples(
        likelihood=connect_likelihood_to_model(
            likelihood.likelihood, masking.sky_with_filter
        ),
        kl_settings=masking.adjust_kl_settings(kl_settings, before_masking=True),
        starting_samples=starting_samples,
        not_take_starting_pos_keys=not_take_starting_pos_keys,
    )

    likelihood = masking_hot_pixels(
        likelihood,
        likelihood.plotting,
        samples,
        mask_hot_pixel=masking,
    )

    return minimization_from_initial_samples(
        likelihood=connect_likelihood_to_model(
            likelihood.likelihood, masking.sky_with_filter
        ),
        kl_settings=masking.adjust_kl_settings(kl_settings, before_masking=False),
        starting_samples=samples,
    )
