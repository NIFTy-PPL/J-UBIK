from collections import namedtuple
from dataclasses import dataclass, field, asdict
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
    _get_std_from_inversestdmodel,
    plot_data_data_model_residuals,
    get_shift_rotation_correction,
)
from ..likelihood.likelihood import (
    GaussianLikelihoodBuilder,
    VariableCovarianceGaussianLikelihoodBuilder,
)


# Define the namedtuple
FilterData = namedtuple("FilterData", ["data", "std", "mask", "builder"])


@dataclass
class ResidualPlottingInformation:
    filter: list[str] = field(default_factory=list)
    data: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    std: list[np.ndarray] = field(default_factory=list)
    builder: list[
        GaussianLikelihoodBuilder | VariableCovarianceGaussianLikelihoodBuilder
    ] = field(default_factory=list)
    y_offset: int = 0

    def append_information(
        self,
        filter: str,
        data: np.ndarray,
        mask: np.ndarray,
        std: np.ndarray,
        builder: GaussianLikelihoodBuilder
        | VariableCovarianceGaussianLikelihoodBuilder,
    ):
        self.filter.append(filter)
        self.data.append(data)
        self.std.append(std)
        self.mask.append(mask)
        self.builder.append(builder)

    def get_filter(self, filter: str) -> FilterData:
        """FilterData for `filter`.

        Parameters
        ----------
        filter: str
            The name of the filter

        Returns
        -------
        FilterData = (data, std, mask, builder)
        """
        index = self.filter.index(filter)
        return FilterData(
            data=self.data[index],
            std=self.std[index],
            mask=self.mask[index],
            builder=self.builder[index],
        )


def _determine_xlen_residuals(
    residual_plotting_info: ResidualPlottingInformation, xmax_residuals: int
):
    maximum = max([d.shape[0] for d in residual_plotting_info.data])
    if maximum > xmax_residuals:
        return xmax_residuals
    return maximum  # because 0 is already there and will not be counted


def _determine_ypos(
    filter_key: str, filter_projector: FilterProjector, y_offset: int = 0
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
    return filter_projector.keys_and_index[filter_key] - y_offset


def build_plot_sky_residuals(
    results_directory: str | None,
    filter_projector: FilterProjector,
    residual_plotting_info: ResidualPlottingInformation,
    sky_model_with_filters: jft.Model,
    residual_plotting_config: ResidualPlottingConfig = ResidualPlottingConfig(),
):
    if results_directory is not None:
        residual_directory = join(results_directory, "residuals")
        makedirs(residual_directory, exist_ok=True)

    xmax_residuals = residual_plotting_config.xmax_residuals

    ylen = len(sky_model_with_filters.target)
    xlen = 3 + _determine_xlen_residuals(residual_plotting_info, xmax_residuals)

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
            residual_plotting_config.sky.get_max(
                np.max(list(tree.map(np.max, sky).values()))
            ),
            residual_plotting_config.sky.get_min(
                np.min(list(tree.map(np.min, sky).values()))
            ),
        )

        for filter_name in filter_projector.keys_and_index.keys():
            ypos = _determine_ypos(
                filter_name, filter_projector, y_offset=residual_plotting_info.y_offset
            )
            axes[ypos, 0].set_title(f"Sky {filter_name}")
            ims[ypos, 0] = axes[ypos, 0].imshow(
                sky[filter_name],
                norm=residual_plotting_config.sky.norm,
                vmax=sky_max,
                vmin=sky_min,
                **residual_plotting_config.sky.rendering,
            )

        overplot_model_sky = (
            None
            if residual_plotting_config.residual_overplot is None
            else _get_model_samples_or_position(
                position_or_samples,
                residual_plotting_config.residual_overplot.overplot_model,
            )
        )

        for filter_key in residual_plotting_info.filter:
            ypos = _determine_ypos(
                filter_key, filter_projector, y_offset=residual_plotting_info.y_offset
            )

            data, std, mask, builder = residual_plotting_info.get_filter(filter_key)

            # TODO : THIS is not quite correct since res**2/std**2 is not linear in std
            if hasattr(builder, "inverse_std_builder"):
                std_new = np.zeros(std.shape)
                std_new[mask] = _get_std_from_inversestdmodel(
                    position_or_samples,
                    inverse_std=builder.inverse_std_builder.build(std=std, mask=mask),
                )
                std = std_new

            model_mean, (redchi_mean, redchi_std) = _get_data_model_and_chi2(
                position_or_samples,
                sky_or_skies,
                data_model=builder.response,
                data=data,
                mask=mask,
                std=std,
            )
            chis = [
                "\n".join((f"redChi2: {mean:.2f} +/- {std:.2f}",))
                for mean, std in zip(redchi_mean, redchi_std)
            ]
            model_mean[~mask] = np.nan

            if len(data.shape) == 2:
                data = [data]
                std = [std]

            ims[ypos, 1:] = plot_data_data_model_residuals(
                ims[ypos, 1:],
                axes[ypos, 1:],
                data_key=filter_key,
                data=data[0],
                data_model=model_mean[0],
                std=std[0],
                residual_over_std=residual_plotting_config.residual_over_std,
                residual_config=residual_plotting_config.residual,
                plotting_config=residual_plotting_config.data,
            )
            display_text(axes[ypos, 3], chis[0])

            for xpos_residual, (data_i, model_i, std_i) in enumerate(
                zip(data[1:], model_mean[1:], std[1:]), start=4
            ):
                if xpos_residual > xlen - 1:
                    continue

                if residual_plotting_config.residual_over_std:
                    axes[ypos, xpos_residual].set_title("(Data - Data model) / std")
                else:
                    axes[ypos, xpos_residual].set_title("Data - Data model")
                    std = 1.0

                ims[ypos, xpos_residual] = axes[ypos, xpos_residual].imshow(
                    (data_i - model_i) / std_i,
                    **asdict(residual_plotting_config.residual),
                    **residual_plotting_config.residual.rendering,
                )
                display_text(axes[ypos, xpos_residual], chis[xpos_residual - 3])

            if residual_plotting_config.residual_overplot is not None:
                overplot_mean, *_ = (
                    (model_mean, None)
                    if overplot_model_sky is None
                    else _get_data_model_and_chi2(
                        position_or_samples,
                        overplot_model_sky,
                        data_model=builder.response,
                        data=data,
                        mask=mask,
                        std=std,
                    )
                )
                overplot_mean[~mask] = np.nan
                for ax, mm in zip(axes[ypos, 3:], overplot_mean):
                    ax.contour(
                        mm / np.nanmax(mm),
                        levels=residual_plotting_config.residual_overplot.max_percent_contours,
                        **residual_plotting_config.residual_overplot.contour_settings,
                    )

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()

        if state_or_none is None:
            plt.show()
        else:
            fig.savefig(
                join(residual_directory, f"{state_or_none.nit:02d}.png"),
                dpi=300,
            )
            plt.close()

    return sky_residuals
