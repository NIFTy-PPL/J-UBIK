from typing import Callable

import nifty.re as jft

from ..parse.plotting import FieldPlottingConfig
from ..psf.psf_operator import PsfDynamic
from .alignment import (
    MultiFilterAlignmentPlottingInformation,
    build_additional,
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

    additional_stuff = []

    if False:
        additional_stuff = [
            build_additional(
                results_directory,
                filter_alignment_data=plotting_alignment_filter,
                plotting_config=plotting_config,
                attribute=lambda model, x: model.sky_model(x),
                name="sky_model",
            )
            for plotting_alignment_filter in plotting_alignment
        ]

    if isinstance(plotting_alignment[0].model[0].psf, PsfDynamic):
        psf_model = [
            build_additional(
                results_directory,
                filter_alignment_data=plotting_alignment_filter,
                plotting_config=plotting_config,
                attribute=lambda model, x: model.psf.model(x),
                name="psf_model",
            )
            for plotting_alignment_filter in plotting_alignment
        ]
        additional_stuff = additional_stuff + psf_model

    def plot_alignment_residuals(
        position_or_samples: dict | jft.Samples,
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        for filter_plot in filters:
            filter_plot(position_or_samples, state_or_none)
        for additional in additional_stuff:
            additional(position_or_samples, state_or_none)

    return plot_alignment_residuals
