from typing import Callable

import nifty.re as jft

from ..parse.plotting import FieldPlottingConfig
from .alignment import (
    MultiFilterAlignmentPlottingInformation,
    build_plot_filter_alignment,
)


def build_plot_alignment_residuals(
    results_directory: str,
    plotting_alignment: MultiFilterAlignmentPlottingInformation,
    plotting_config: FieldPlottingConfig = FieldPlottingConfig(),
    name_append: str = "",
    interactive: bool = False,
) -> Callable[dict | jft.Samples | jft.Vector, None]:
    filters = [
        build_plot_filter_alignment(
            results_directory,
            filter_alignment_data=plotting_alignment_filter,
            plotting_config=plotting_config,
            name_append=name_append,
            interactive=interactive,
        )
        for plotting_alignment_filter in plotting_alignment
    ]

    def plot_alignment_residuals(
        position_or_samples: dict | jft.Samples,
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        for filter_plot in filters:
            filter_plot(position_or_samples, state_or_none)

    return plot_alignment_residuals
