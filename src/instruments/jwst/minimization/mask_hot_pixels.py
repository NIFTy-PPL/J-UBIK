from typing import Optional
import os
from dataclasses import dataclass, asdict, field

import scipy
import numpy as np
import nifty8.re as jft

from .mask_hot_pixel_data import HotPixelMaskingData
from ....likelihood import connect_likelihood_to_model
from ..likelihood.target_likelihood import (
    build_target_likelihood,
    TargetLikelihoodSideEffects,
)
from ..data.loader.target_loader import TargetDataCore

from ....minimization_parser import MinimizationParser
from ....minimization.minimization_from_samples import (
    KLSettings,
    minimization_from_initial_samples,
)
from ..jwst_likelihoods import TargetLikelihoodProducts
from ..plotting.residuals import ResidualPlottingInformation


class HotPixelMasking:
    def __init__(
        self,
        yaml_dict: dict,
        sky_with_filter: jft.Model,
        hot_pixel_masking_data: HotPixelMaskingData,
        star_hot_pixel: float,
        star_sigma: float,
        res_dir: str,
    ) -> None:
        self.mask_at_step = yaml_dict["hot_pixel"]["mask_at_step"]
        self.sigma = yaml_dict["hot_pixel"]["sigma"]
        self.sky_with_filter = sky_with_filter
        self.hot_pixel_masking_data = hot_pixel_masking_data
        self.hot_star_sigma = star_hot_pixel
        self.star_sigma = star_sigma

        self.res_dir = os.path.join(res_dir, "masking")
        os.makedirs(self.res_dir, exist_ok=True)

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


def _hot_pixel_mask(res, mask_hot_pixel: HotPixelMasking):
    return np.abs(res) < mask_hot_pixel.sigma


def _hot_star_mask_from_nanpixel(res, m, filter_name, mask_hot_pixel: HotPixelMasking):
    import matplotlib.pyplot as plt

    res_var = np.zeros(m.shape)
    res_var[m] = res

    jj = mask_hot_pixel.hot_pixel_masking_data.filter.index(filter_name)
    mnan = ~mask_hot_pixel.hot_pixel_masking_data.nan_mask[jj]

    mask = m.copy()

    for kk, ii, jj in zip(*np.where(mnan)):
        try:
            res00 = res_var[kk, ii - 1, jj - 1]
            res01 = res_var[kk, ii - 1, jj]
            res02 = res_var[kk, ii - 1, jj + 1]

            res10 = res_var[kk, ii, jj - 1]
            res11 = res_var[kk, ii, jj]
            res12 = res_var[kk, ii, jj + 1]

            res20 = res_var[kk, ii + 1, jj - 1]
            res21 = res_var[kk, ii + 1, jj]
            res22 = res_var[kk, ii + 1, jj + 1]

            arr = np.sqrt(
                np.array(
                    [
                        [res00, res01, res02],
                        [res10, res11, res12],
                        [res20, res21, res22],
                    ],
                )
                ** 2
            )

            if arr.sum() == 0:
                continue

            evaluate = np.sqrt((res01**2 + res10**2 + res12**2 + res21**2) / 4)
            if evaluate > mask_hot_pixel.hot_star_sigma:
                mask[kk, ii - 1, jj] = False
                mask[kk, ii, jj - 1] = False
                mask[kk, ii, jj + 1] = False
                mask[kk, ii + 1, jj] = False

        except IndexError:
            pass

    return mask[m]


def _hot_star_mask_convolution(res, m, filter_name, mask_hot_pixel: HotPixelMasking):
    import matplotlib.pyplot as plt

    res_var = np.zeros(m.shape)
    res_var[m] = res

    star = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    eval_field = np.sqrt(res_var**2)

    convolved_res = np.array([scipy.ndimage.convolve(ev, star) for ev in eval_field])
    mask = m.copy()

    for kk, ii, jj in zip(*np.where(convolved_res > mask_hot_pixel.star_sigma)):
        mask[kk, ii, jj] = False
        mask[kk, ii - 1, jj] = False
        mask[kk, ii, jj - 1] = False
        mask[kk, ii, jj + 1] = False
        mask[kk, ii + 1, jj] = False

    return mask[m]


@dataclass
class MaskPlotting:
    res_dir: str
    filter: str


def masking_plot(res, mm_orig, m_hot, m_nan, m_star, m_tot, plotting: MaskPlotting):
    import matplotlib.pyplot as plt

    # PLOTTING
    res_var = np.zeros(mm_orig.shape)
    res_var[mm_orig] = res
    evfield = np.sqrt(res_var**2)

    mm_hot = mm_orig.copy()
    mm_hot[mm_hot] = m_hot

    mm_nan = mm_hot.copy()
    mm_nan[mm_nan] = m_nan

    mm_star = mm_hot.copy()
    mm_star[mm_star] = m_star

    m_tot = m_hot.copy()
    m_tot[m_hot] = m_star * m_nan

    mm_tot = mm_orig.copy()
    mm_tot[mm_tot] = m_tot

    fig, axes = plt.subplots(len(evfield), 6, sharex=True, sharey=True, dpi=300)
    for i, (axi, ev, orig, hot, nan, star, tot) in enumerate(
        zip(axes, evfield, mm_orig, mm_hot, mm_nan, mm_star, mm_tot)
    ):
        a0, a1, a2, a3, a4, a5 = axi
        a0.imshow(ev, origin="lower")
        a1.imshow(orig, origin="lower", cmap="binary_r")
        a2.imshow(hot, origin="lower", cmap="binary_r")
        a3.imshow(nan, origin="lower", cmap="binary_r")
        a4.imshow(star, origin="lower", cmap="binary_r")
        a5.imshow(tot, origin="lower", cmap="binary_r")
        if i == 0:
            a0.set_title("residuals")
            a1.set_title("orig")
            a2.set_title("hot")
            a3.set_title("nan")
            a4.set_title("star")
            a5.set_title("tot")
    plt.tight_layout()
    plt.savefig(f"{os.path.join(plotting.res_dir, plotting.filter)}.png")
    plt.close()


def _build_new_mask_and_response(
    dm, d, m, s, R, mask_hot_pixel: HotPixelMasking, filter_name: str
):
    res = (d[m] - dm) / s[m]
    m_hot = _hot_pixel_mask(res, mask_hot_pixel)

    mm = m.copy()
    mm[m] = m_hot

    m_nan = _hot_star_mask_from_nanpixel(res[m_hot], mm, filter_name, mask_hot_pixel)
    m_star = _hot_star_mask_convolution(res[m_hot], mm, filter_name, mask_hot_pixel)

    m_tot = m_hot.copy()
    m_tot[m_hot] = m_nan  # * m_star
    mm_tot = m.copy()
    mm_tot[mm_tot] = m_tot

    masking_plot(
        res,
        m,
        m_hot,
        m_nan,
        m_star,
        m_tot,
        MaskPlotting(mask_hot_pixel.res_dir, filter=filter_name),
    )

    return jft.Model(lambda x: R(x)[m_tot], domain=R.domain), mm_tot


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
        dm = jft.mean([response(si, ll.builder.response) for si in samples])
        response_new, mask_3d = _build_new_mask_and_response(
            dm,
            ll.builder.data,
            ll.builder.mask,
            ll.builder.std,
            ll.builder.response,
            mask_hot_pixel,
            ll.filter,
        )

        new_likelihoods.append(
            build_target_likelihood(
                response=response_new,
                target_data=TargetDataCore(
                    data=ll.builder.data,
                    std=ll.builder.std,
                    mask=mask_3d,
                ),
                filter_name=ll.filter,
                side_effect=TargetLikelihoodSideEffects(plotting=target_plotting),
            )
        )

        # TODO: MAKE THIS NECESSERY SIDE EFFECT DISAPPEAR
        plotting.mask[plotting.filter.index(ll.filter)] = mask_3d
        plotting.model[plotting.filter.index(ll.filter)] = response_new

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
        likelihood, likelihood.plotting, samples, mask_hot_pixel=masking
    )

    return minimization_from_initial_samples(
        likelihood=connect_likelihood_to_model(
            likelihood.likelihood, masking.sky_with_filter
        ),
        kl_settings=masking.adjust_kl_settings(kl_settings, before_masking=False),
        starting_samples=None,
    )
