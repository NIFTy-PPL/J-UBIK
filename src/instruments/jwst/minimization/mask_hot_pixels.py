import os
from dataclasses import asdict, dataclass
from typing import Optional

from numpy.typing import NDArray

import nifty.re as jft
import numpy as np
import scipy

from ....likelihood import connect_likelihood_to_model
from ....minimization.minimization_from_samples import (
    KLSettings,
    minimization_from_initial_samples,
)
from ....minimization_parser import MinimizationParser
from ..data.loader.target_loader import TargetDataCore
from ..jwst_likelihoods import TargetLikelihoodProducts
from ..likelihood.target_likelihood import build_target_likelihood
from ..parse.minimization.mask_hot_pixels import MaskingStepSettings
from ..plotting.residuals import ResidualPlottingInformation
from .mask_hot_pixel_data import HotPixelMaskingData


class MaskingStep:
    def __init__(
        self,
        settings: MaskingStepSettings,
        sky_with_filter: jft.Model,
        hot_pixel_masking_data: HotPixelMaskingData,
        res_dir: str,
    ) -> None:
        self.settings = settings
        self.sky_with_filter = sky_with_filter
        self.hot_pixel_masking_data = hot_pixel_masking_data

        self.res_dir = os.path.join(res_dir, "masking_step")
        os.makedirs(self.res_dir, exist_ok=True)

    def adjust_kl_settings(
        self, kl_settings: KLSettings, before_masking: bool = False
    ) -> KLSettings:
        if before_masking:
            start = 0
            end = self.settings.mask_at_iteration
            resume = kl_settings.resume
        else:
            start = self.settings.mask_at_iteration
            end = kl_settings.n_total_iterations
            resume = True

        klset = asdict(kl_settings)
        klset["minimization"] = MinimizationParserUpdate.minimization_at_iteration(
            kl_settings.minimization, start
        )
        klset["n_total_iterations"] = end
        klset["resume"] = resume

        return KLSettings(**klset)


def _hot_pixel_mask(residual: np.ndarray, threshold_hot_pixel: float):
    """This process masks pixels if their absolute residual is above the
    `threshold_hot_pixel` threshold.

    Parameters
    ----------
    residual: np.ndarray
        The residuals array.
    threshold_hot_pixel: float
        The threshold for masking the hot pixels.
    """
    if threshold_hot_pixel is None:
        return True
    return np.abs(residual) < threshold_hot_pixel


def _hot_star_mask_from_nanpixel(
    residual: np.ndarray,
    mask_3d_og: np.ndarray,
    mask_3d_nan: np.ndarray,
    threshold_star_nan: float | None,
) -> NDArray:
    """This process masks neighbors of `nan`-pixel (masked by the JWST pipeline), if the
    summed, squared residuals of the pixels are above the `threshold_star_nan` threshold.

    Parameters
    ----------
    residual: np.ndarray
        The residuals array
    mask_3d_og: np.ndarray
        The original mask
    mask_3d_nan: np.ndarray
        The original mask that contains the `nan` values from the JWST pipeline.
    threshold_star_nan: float
        The threshold for masking the neighboring `star` pixels. If `None` the process
        is switched off.

    Returns
    -------
    NDArray: New mask
    """

    if threshold_star_nan is None:
        return True

    res_var = np.zeros(mask_3d_og.shape)
    res_var[mask_3d_og] = residual

    mask_3d_tmp = mask_3d_og.copy()

    for kk, ii, jj in zip(*np.where(~mask_3d_nan)):
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
            if evaluate > threshold_star_nan:
                mask_3d_tmp[kk, ii - 1, jj] = False
                mask_3d_tmp[kk, ii, jj - 1] = False
                mask_3d_tmp[kk, ii, jj + 1] = False
                mask_3d_tmp[kk, ii + 1, jj] = False

        except IndexError:
            # NOTE : This is handling the edges of the field, where the stars fall
            # outside the grid. Typically these are masked, anyways.
            pass

    return mask_3d_tmp[mask_3d_og]


def _hot_star_mask_convolution(
    residual: np.ndarray,
    mask_3d_og: np.ndarray,
    threshold_star_convolution: float | None,
):
    """This process masks pixels if by convolving the squared residuals with a star mask
    falls above the `threshold_star_convolution` threshold.

    Parameters
    ----------
    residual: np.ndarray
        The residuals array.
    mask_3d_og: np.ndarray
        The original mask to be updated.
    threshold_star_convolution: float | None
        The threshold for masking the hot star pixels. If `None` the process is switched
        off.
    """

    if threshold_star_convolution is None:
        return True

    res_var = np.zeros(mask_3d_og.shape)
    res_var[mask_3d_og] = residual

    star = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    eval_field = np.sqrt(res_var**2)

    convolved_res = np.array([scipy.ndimage.convolve(ev, star) for ev in eval_field])
    mask = mask_3d_og.copy()

    for kk, ii, jj in zip(*np.where(convolved_res > threshold_star_convolution)):
        mask[kk, ii, jj] = False
        mask[kk, ii - 1, jj] = False
        mask[kk, ii, jj - 1] = False
        mask[kk, ii, jj + 1] = False
        mask[kk, ii + 1, jj] = False

    return mask[mask_3d_og]


@dataclass
class MaskPlotting:
    res_dir: str
    filter: str


def masking_plot(
    residual,
    mm_orig,
    mask_flat_hot_pixel,
    mask_flat_nanstar,
    mask_flat_hotstar,
    mask_flat_total,
    plotting: MaskPlotting,
):
    import matplotlib.pyplot as plt

    # PLOTTING
    res_var = np.zeros(mm_orig.shape)
    res_var[mm_orig] = residual
    evfield = np.sqrt(res_var**2)

    mm_hot = mm_orig.copy()
    mm_hot[mm_hot] = mask_flat_hot_pixel

    mm_nan = mm_hot.copy()
    mm_nan[mm_nan] = mask_flat_nanstar

    mm_star = mm_hot.copy()
    mm_star[mm_star] = mask_flat_hotstar

    mask_flat_total = mask_flat_hot_pixel.copy()
    mask_flat_total[mask_flat_hot_pixel] = mask_flat_hotstar * mask_flat_nanstar

    mask_3d_total = mm_orig.copy()
    mask_3d_total[mask_3d_total] = mask_flat_total

    fig, axes = plt.subplots(len(evfield), 6, sharex=True, sharey=True, dpi=300)
    for i, (axi, ev, orig, hot, nan, star, tot) in enumerate(
        zip(axes, evfield, mm_orig, mm_hot, mm_nan, mm_star, mask_3d_total)
    ):
        a0, a1, a2, a3, a4, a5 = axi
        a0.imshow(ev, origin="lower", interpolation="None")
        a1.imshow(orig, origin="lower", cmap="binary_r", interpolation="None")
        a2.imshow(hot, origin="lower", cmap="binary_r", interpolation="None")
        a3.imshow(nan, origin="lower", cmap="binary_r", interpolation="None")
        a4.imshow(star, origin="lower", cmap="binary_r", interpolation="None")
        a5.imshow(tot, origin="lower", cmap="binary_r", interpolation="None")
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


@dataclass
class MaskingProducts:
    mask_flat: np.ndarray
    mask_3d: np.ndarray


def _build_new_mask_strategy(
    model_data_mean: np.ndarray,
    data: np.ndarray,
    mask_3d_og: np.ndarray,
    std: np.ndarray,
    masking_step: MaskingStep,
    filter_name: str,
):
    residual = (data[mask_3d_og] - model_data_mean) / std[mask_3d_og]
    mask_flat_hot_pixel = _hot_pixel_mask(
        residual, masking_step.settings.threshold_hot_pixel
    )

    mask_3d_tmp = mask_3d_og.copy()
    mask_3d_tmp[mask_3d_og] = mask_flat_hot_pixel

    mask_flat_nanstar = _hot_star_mask_from_nanpixel(
        residual[mask_flat_hot_pixel],
        mask_3d_tmp,
        masking_step.hot_pixel_masking_data.get_filter_nanmask(filter_name),
        masking_step.settings.threshold_star_nan,
    )
    mask_flat_hotstar = _hot_star_mask_convolution(
        residual[mask_flat_hot_pixel],
        mask_3d_tmp,
        masking_step.settings.threshold_star_convolution,
    )

    mask_flat_total = mask_flat_hot_pixel.copy()
    mask_flat_total[mask_flat_hot_pixel] = mask_flat_nanstar * mask_flat_hotstar
    mask_3d_total = mask_3d_og.copy()
    mask_3d_total[mask_3d_total] = mask_flat_total

    masking_plot(
        residual,
        mask_3d_og,
        mask_flat_hot_pixel,
        mask_flat_nanstar,
        mask_flat_hotstar,
        mask_flat_total,
        MaskPlotting(masking_step.res_dir, filter=filter_name),
    )

    return MaskingProducts(
        mask_flat=mask_flat_total,
        mask_3d=mask_3d_total,
    )


def masking_hot_pixels(
    likelihood: TargetLikelihoodProducts,
    plotting: ResidualPlottingInformation,
    samples: jft.Samples,
    masking_step: MaskingStep,
) -> TargetLikelihoodProducts:
    def tmp_response(si, R):
        return R(masking_step.sky_with_filter(si) | si.tree)

    target_plotting = ResidualPlottingInformation(y_offset=plotting.y_offset)

    new_likelihoods = []
    for ll in likelihood.likelihoods:
        model_data_mean = jft.mean(
            [tmp_response(si, ll.builder.response) for si in samples]
        )
        masking_products = _build_new_mask_strategy(
            model_data_mean,
            ll.builder.data,
            ll.builder.mask,
            ll.builder.std,
            masking_step,
            ll.filter,
        )

        inv_std = None
        if hasattr(ll.builder, "inverse_std_builder"):
            inv_std = ll.builder.inverse_std_builder

        response = jft.Model(
            lambda x: ll.builder.response(x)[masking_products.mask_flat],
            domain=ll.builder.response.domain,
        )

        new_likelihoods.append(
            build_target_likelihood(
                response=response,
                target_data=TargetDataCore(
                    data=ll.builder.data,
                    std=ll.builder.std,
                    mask=masking_products.mask_3d,
                ),
                filter_name=ll.filter,
                inverse_std_builder=inv_std,
            )
        )

        # TODO: MAKE THIS NECESSERY SIDE EFFECT DISAPPEAR
        plotting.mask[plotting.filter.index(ll.filter)] = masking_products.mask_3d
        plotting.model[plotting.filter.index(ll.filter)] = response

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
    masking_step: MaskingStep,
    starting_samples: Optional[jft.Samples] = None,
    not_take_starting_pos_keys: tuple[str] = (),
):
    samples, state = minimization_from_initial_samples(
        likelihood=connect_likelihood_to_model(
            likelihood.likelihood, masking_step.sky_with_filter
        ),
        kl_settings=masking_step.adjust_kl_settings(kl_settings, before_masking=True),
        starting_samples=starting_samples,
        not_take_starting_pos_keys=not_take_starting_pos_keys,
    )

    likelihood = masking_hot_pixels(
        likelihood, likelihood.plotting, samples, masking_step=masking_step
    )

    return minimization_from_initial_samples(
        likelihood=connect_likelihood_to_model(
            likelihood.likelihood, masking_step.sky_with_filter
        ),
        kl_settings=masking_step.adjust_kl_settings(kl_settings, before_masking=False),
        starting_samples=None,
    )
