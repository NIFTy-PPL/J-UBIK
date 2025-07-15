import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import nifty.re as jft
import matplotlib.pyplot as plt

from ..jwst_response import JwstResponse
from ..parse.plotting import FieldPlottingConfig, ResidualPlottingConfig
from .plotting_base import (
    display_text,
    _get_data_model_and_chi2,
    _get_model_samples_or_position,
    plot_data_data_model_residuals,
)

# Define the namedtuple
StarDataPlotting = namedtuple(
    "StarDataPlotting", ["data", "std", "mask", "model", "subsample"]
)


@dataclass
class FilterAlignmentPlottingInformation:
    filter: str
    subsample: list[int] = field(default_factory=list)
    star_id: list[int] = field(default_factory=list)
    data: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    std: list[np.ndarray] = field(default_factory=list)
    model: list[JwstResponse] = field(default_factory=list)

    def append_information(
        self,
        star_id: int,
        subsample: int,
        data: np.ndarray,
        mask: np.ndarray,
        std: np.ndarray,
        model: JwstResponse,
    ) -> None:
        self.star_id.append(star_id)
        self.subsample.append(subsample)
        self.data.append(data)
        self.std.append(std)
        self.mask.append(mask)
        self.model.append(model)

    def get_star(self, star_id: int) -> StarDataPlotting:
        """StarDataPlotting for `filter`.

        Parameters
        ----------
        filter: str
            The name of the filter

        Returns
        -------
        StarDataPlotting = (data, std, mask, model)
        """
        index = self.star_id.index(star_id)
        return StarDataPlotting(
            data=self.data[index],
            std=self.std[index],
            mask=self.mask[index],
            model=self.model[index],
            subsample=self.subsample[index],
        )


@dataclass
class MultiFilterAlignmentPlottingInformation:
    """This is

    psf:
        The FilterAlignmentPlottingInformation for the pipeline psf model.
    convolved:
        The FilterAlignmentPlottingInformation for a convolved psf model.
    """

    psf: list[FilterAlignmentPlottingInformation] = field(default_factory=list)
    convolved: list[FilterAlignmentPlottingInformation] = field(default_factory=list)


def build_additional(
    results_directory: str,
    filter_alignment_data: FilterAlignmentPlottingInformation,
    plotting_config: FieldPlottingConfig = FieldPlottingConfig(vmin=1e-4, norm="log"),
    attribute=lambda model, x: model.sky_model(x),
    name="sky_model",
) -> Callable[dict | jft.Samples | jft.Vector, None]:
    extra_directory = os.path.join(results_directory, "alignment_extra")
    os.makedirs(extra_directory, exist_ok=True)

    filter_name = filter_alignment_data.filter
    ylen = len(filter_alignment_data.star_id)
    xlen = max([dd.shape[0] for dd in filter_alignment_data.data])

    def filter_alignment(
        position_or_samples: dict | jft.Samples,
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        fig, axes = plt.subplots(ylen, xlen, figsize=(3 * xlen, 3 * ylen), dpi=300)
        ims = np.zeros_like(axes)
        if ylen == 1:
            ims = ims[None]
            axes = axes[None]

        if not isinstance(position_or_samples, jft.Samples):
            position_or_samples = [position_or_samples]

        for ypos, star_id in enumerate(filter_alignment_data.star_id):
            _, _, _, model = filter_alignment_data.get_star(star_id)
            extra = jft.mean([attribute(model, si) for si in position_or_samples])

            for xpos, ee in enumerate(extra):
                max_d = plotting_config.get_max(ee)
                min_d = plotting_config.get_min(ee)
                ims[ypos, xpos] = axes[ypos, xpos].imshow(
                    ee,
                    vmin=min_d,
                    vmax=max_d,
                    norm=plotting_config.norm,
                    **plotting_config.rendering,
                )

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()

        if state_or_none is None:
            plt.show()
        else:
            fig.savefig(
                os.path.join(
                    extra_directory,
                    f"{name}_{filter_name}_{state_or_none.nit:02d}.png",
                ),
                dpi=300,
            )
            plt.close()

    return filter_alignment


def build_plot_filter_alignment(
    results_directory: str,
    filter_alignment_data: FilterAlignmentPlottingInformation,
    plotting_config: FieldPlottingConfig = FieldPlottingConfig(),
    residual_config: ResidualPlottingConfig = ResidualPlottingConfig(),
    name_append: str = "",
    interactive: bool = False,
) -> Callable[dict | jft.Samples | jft.Vector, None]:
    alignment_directory = os.path.join(results_directory, "alignment")
    os.makedirs(alignment_directory, exist_ok=True)

    filter_name = filter_alignment_data.filter
    ylen = len(filter_alignment_data.star_id)
    xlen = 3 * max([dd.shape[0] for dd in filter_alignment_data.data])

    def filter_alignment(
        position_or_samples: dict | jft.Samples,
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        if state_or_none is not None:
            jft.logger.info(f"Plotting alignment {state_or_none.nit}")

        fig, axes = plt.subplots(ylen, xlen, figsize=(3 * xlen, 3 * ylen), dpi=300)
        ims = np.zeros_like(axes)
        if ylen == 1:
            ims = ims[None]
            axes = axes[None]

        for ypos, star_id in enumerate(filter_alignment_data.star_id):
            data, std, mask, model, subsample = filter_alignment_data.get_star(star_id)

            model_mean, (redchi_mean, redchi_std) = _get_data_model_and_chi2(
                position_or_samples=position_or_samples,
                sky_or_skies=None,
                data_model=model,
                data=data,
                mask=mask,
                std=std,
            )

            if interactive:
                exit()

            chis = [
                "\n".join((f"redChi2: {mean:.2f} +/- {std:.2f}",))
                for mean, std in zip(redchi_mean, redchi_std)
            ]

            star_positions_in_data = _get_model_samples_or_position(
                position_or_samples, model.sky_model.location
            )
            star_positions_in_data = star_positions_in_data / subsample

            for ii in range(len(data)):
                length_xpos = 3
                xpos = ii * length_xpos

                ims[ypos, xpos:] = plot_data_data_model_residuals(
                    ims[ypos, xpos:],
                    axes[ypos, xpos:],
                    data_key=star_id,
                    data=data[ii],
                    data_model=model_mean[ii],
                    std=std[ii],
                    residual_over_std=residual_config.residual_over_std,
                    residual_config=residual_config.residual,
                    plotting_config=plotting_config,
                )

                samples_star_position_in_data = star_positions_in_data[:, ii, :]
                for position in samples_star_position_in_data:
                    for jj in range(length_xpos):
                        axes[ypos, xpos + jj].scatter(
                            *position, marker="x", linewidths=0.1
                        )

                display_text(axes[ypos, xpos + 1], chis[ii])

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()

        if state_or_none is None:
            plt.show()
        else:
            fig.savefig(
                os.path.join(
                    alignment_directory,
                    f"{filter_name}{name_append}_{state_or_none.nit:02d}.png",
                ),
                dpi=300,
            )
            plt.close()

    return filter_alignment
