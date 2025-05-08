from collections import namedtuple
from dataclasses import dataclass, field
from os import makedirs
from os.path import join
from typing import Union

import nifty8.re as jft
import matplotlib.pyplot as plt
import numpy as np
from jax import tree

from ..filter_projector import FilterProjector
from ..parse.plotting import ResidualPlottingConfig
from .plotting_base import (
    display_text,
    _get_model_samples_or_position,
    _get_data_model_and_chi2,
    plot_data_data_model_residuals,
    get_shift_rotation_correction,
)
from ..jwst_response import JwstResponse


# Define the namedtuple
FilterData = namedtuple("FilterData", ["data", "std", "mask", "model"])


@dataclass
class ResidualPlottingInformation:
    filter: list[str] = field(default_factory=list)
    data: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    std: list[np.ndarray] = field(default_factory=list)
    model: list[JwstResponse] = field(default_factory=list)

    def append_information(
        self,
        filter: str,
        data: np.ndarray,
        mask: np.ndarray,
        std: np.ndarray,
        model: JwstResponse,
    ):
        self.filter.append(filter)
        self.data.append(data)
        self.std.append(std)
        self.mask.append(mask)
        self.model.append(model)

    def get_filter(self, filter: str) -> FilterData:
        """FilterData for `filter`.

        Parameters
        ----------
        filter: str
            The name of the filter

        Returns
        -------
        FilterData = (data, std, mask, model)
        """
        index = self.filter.index(filter)
        return FilterData(
            data=self.data[index],
            std=self.std[index],
            mask=self.mask[index],
            model=self.model[index],
        )


def _determine_xlen_residuals(
    residual_plotting_info: ResidualPlottingInformation, xmax_residuals: int
):
    maximum = max([d.shape[0] for d in residual_plotting_info.data])
    if maximum > xmax_residuals:
        return xmax_residuals
    return maximum  # because 0 is already there and will not be counted


def _determine_ypos(
    filter_key: str,
    filter_projector: FilterProjector,
) -> int:
    """Determine the y position of the panel in the residual plot.

    Parameters
    ----------
    dkey: str
        The data_key which will determine the energy bin and hence the position
        on the panel grid.
    filter_projector: FilterProjector
        The filter_projector of the reconstruction.
    ylen_offset: int
        An offset which corresponds to the bins not considered in determining
        the ypos of the panels. I.e. the sky can have more bins than the ones
        plotted in the panels, the ylen_offset corrects for this.

    Returns
    -------
    ypos: int
        The y-position on the panel grid.
    """
    return filter_projector.keys_and_index[filter_key]


def build_plot_sky_residuals(
    results_directory: str,
    filter_projector: FilterProjector,
    residual_plotting_info: ResidualPlottingInformation,
    sky_model_with_filters: jft.Model,
    plotting_config: ResidualPlottingConfig = ResidualPlottingConfig(),
):
    residual_directory = join(results_directory, "residuals")
    makedirs(residual_directory, exist_ok=True)

    display_pointing = plotting_config.display_pointing
    display_chi2 = plotting_config.display_chi2
    std_relative = plotting_config.std_relative
    fileformat = plotting_config.fileformat
    xmax_residuals = plotting_config.xmax_residuals

    residual_plotting_config = plotting_config.data

    ylen = len(sky_model_with_filters.target)
    xlen = 3 + _determine_xlen_residuals(residual_plotting_info, xmax_residuals)

    rendering = plotting_config.sky.rendering

    def sky_residuals(
        position_or_samples: Union[dict, jft.Samples],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        jft.logger.info(f"Results: {results_directory}")
        jft.logger.info("Plotting residuals")

        fig, axes = plt.subplots(ylen, xlen, figsize=(3 * xlen, 3 * ylen), dpi=300)
        ims = np.zeros_like(axes)
        if ylen == 1:
            ims = ims[None]
            axes = axes[None]

        sky_or_skies = _get_model_samples_or_position(
            position_or_samples, sky_model_with_filters
        )

        if isinstance(position_or_samples, jft.Samples):
            sky = jft.mean(sky_or_skies)
        else:
            sky = sky_or_skies
        sky_max, sky_min = (
            plotting_config.sky.get_max(np.max(list(tree.map(np.max, sky).values()))),
            plotting_config.sky.get_min(np.min(list(tree.map(np.min, sky).values()))),
        )

        for filter_name, ypos in filter_projector.keys_and_index.items():
            axes[ypos, 0].set_title(f"Sky {filter_name}")
            ims[ypos, 0] = axes[ypos, 0].imshow(
                sky[filter_name],
                norm=plotting_config.sky.norm,
                vmax=sky_max,
                vmin=sky_min,
                **rendering,
            )

        for filter_key in residual_plotting_info.filter:
            ypos = filter_projector.keys_and_index[filter_key]

            data, std, mask, model = residual_plotting_info.get_filter(filter_key)

            model_mean, (redchi_mean, redchi_std) = _get_data_model_and_chi2(
                position_or_samples,
                sky_or_skies,
                data_model=model,
                data=data,
                mask=mask,
                std=std,
            )
            chis = [
                "\n".join((f"redChi2: {mean:.2f} +/- {std:.2f}",))
                for mean, std in zip(redchi_mean, redchi_std)
            ]

            if len(data.shape) == 2:
                data = [data]
                std = [std]

            ims[ypos, 1:] = plot_data_data_model_residuals(
                ims[ypos, 1:],
                axes[ypos, 1:],
                data_key=filter_key,
                data=data[0],
                data_model=model_mean[0],
                std=std[0] if std_relative else 1.0,
                plotting_config=residual_plotting_config,
            )
            display_text(axes[ypos, 3], chis[0])

            if display_pointing:
                (sh_m, sh_s), (ro_m, ro_s) = get_shift_rotation_correction(
                    position_or_samples,
                    model.rotation_and_shift.correction_model,
                )
                data_model_text = "\n".join(
                    (
                        f"dx={sh_m[0]:.1e}+-{sh_s[0]:.1e}",
                        f"dy={sh_m[1]:.1e}+-{sh_s[1]:.1e}",
                        f"dth={ro_m:.1e}+-{ro_s:.1e}",
                    )
                )
                display_text(axes[ypos, 2], data_model_text)

            for xpos_residual, (data_i, model_i, std_i) in enumerate(
                zip(data[1:], model_mean[1:], std[1:]), start=4
            ):
                if xpos_residual > xlen - 1:
                    continue

                ims[ypos, xpos_residual] = axes[ypos, xpos_residual].imshow(
                    (data_i - model_i) / std_i,
                    vmin=-3,
                    vmax=3,
                    cmap="RdBu_r",
                    **rendering,
                )
                display_text(axes[ypos, xpos_residual], chis[xpos_residual - 3])

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()

        if state_or_none is None:
            plt.show()
        else:
            fig.savefig(
                join(residual_directory, f"{state_or_none.nit:02d}.{fileformat}"),
                dpi=300,
            )
            plt.close()

    return sky_residuals
