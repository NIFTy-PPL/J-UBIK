from dataclasses import dataclass, field
from os import makedirs
from os.path import join
from typing import Union

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np

from ....grid import Grid
from .plotting_base import get_alpha_and_reference, get_position_or_samples_of_model
from ..parse.plotting import FieldPlottingConfig, MultiFrequencyPlottingConfig


def build_plot_source(
    results_directory: str,
    plotting_config: MultiFrequencyPlottingConfig,
    lens_system,
    grid: Grid,
):
    """

    plotting_config:
        norm_source: Normalize (default)
        min_source: 1e-5 (default)
        extent: None (default)
    """

    from charm_lensing.lens_system import LensSystem

    lens_system: LensSystem = lens_system

    if results_directory is not None:
        source_dir = join(results_directory, "source")
        makedirs(source_dir, exist_ok=True)

    freq_len = len(grid.spectral)
    xlen = 3
    ylen = 1 + int(np.ceil(freq_len / xlen))

    source_light = lens_system.source_plane_model.light_model
    source_light_parametric = lens_system.source_plane_model.light_model.parametric
    source_light_alpha, source_light_reference = get_alpha_and_reference(
        lens_system.source_plane_model.light_model
    )

    models = [
        source_light,
        source_light_parametric,
        source_light_alpha,
        source_light_reference,
    ]

    for ii, model in enumerate(models):
        if model is None:
            models[ii] = lambda _: np.zeros((2, 2))

    filter_plotting_config = plotting_config.combined
    reference_plotting_config: FieldPlottingConfig = plotting_config.reference
    spectral_index_plotting_config: FieldPlottingConfig = plotting_config.alpha

    rendering = filter_plotting_config.rendering
    rendering["extent"] = lens_system.source_plane_model.space.extend().extent

    def plot_source(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        print("Plotting source light")

        sl, sl_para, sl_alpha, sl_ref = [
            get_position_or_samples_of_model(position_or_samples, model, True)
            for model in models
        ]
        if len(sl_para.shape) > 2:
            sl_para = sl_para.mean(axis=0)

        vmin, vmax = filter_plotting_config.get_min(sl), None

        fig, axes = plt.subplots(ylen, xlen, figsize=(3 * xlen, 3 * ylen), dpi=300)
        ims = np.zeros_like(axes)

        # Plot lens light
        axes[0, 0].set_title("Parametric model")
        axes[0, 1].set_title("Reference model at I0")
        axes[0, 2].set_title("Spectral index")
        ims[0, 0] = axes[0, 0].imshow(sl_para, **rendering)
        ims[0, 1] = axes[0, 1].imshow(
            sl_ref,
            vmin=reference_plotting_config.get_min(sl_ref),
            vmax=reference_plotting_config.get_max(sl_ref),
            norm=reference_plotting_config.norm,
            **rendering,
        )
        ims[0, 2] = axes[0, 2].imshow(
            sl_alpha,
            vmin=spectral_index_plotting_config.get_min(sl_alpha),
            vmax=spectral_index_plotting_config.get_max(sl_alpha),
            norm=spectral_index_plotting_config.norm,
            **rendering,
        )

        axes = axes.flatten()
        ims = ims.flatten()
        for ii, (energy_range, fld) in enumerate(zip(grid.spectral.color_ranges, sl)):
            energy, energy_unit = energy_range.center.value, energy_range.center.unit
            ii += 3

            axes[ii].set_title(f"{energy:.4f} {energy_unit}")
            ims[ii] = axes[ii].imshow(
                fld, norm=filter_plotting_config.norm, vmin=vmin, vmax=vmax, **rendering
            )

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                try:
                    fig.colorbar(im, ax=ax, shrink=0.7)
                except ValueError:
                    pass

        if state_or_none is not None:
            fig.tight_layout()
            fig.savefig(
                join(source_dir, f"source_{state_or_none.nit:02d}.png"), dpi=300
            )
            plt.close()
        else:
            plt.show()

    return plot_source
