from dataclasses import dataclass, field
from os import makedirs
from os.path import join
from typing import Union

import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np

from ....grid import Grid
from .plotting_base import get_alpha_and_reference, get_position_or_samples_of_model
from ..parse.plotting import FieldPlottingConfig


def build_plot_source(
    results_directory: str,
    plotting_config: FieldPlottingConfig,
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

    rendering = plotting_config.rendering
    rendering["extent"] = lens_system.source_plane_model.space.extend().extent

    def plot_source(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        print("Plotting source light")

        sl, slp, sla, sln = [
            get_position_or_samples_of_model(position_or_samples, model, True)
            for model in models
        ]

        vmin, vmax = plotting_config.get_min(sl), None

        fig, axes = plt.subplots(ylen, xlen, figsize=(3 * xlen, 3 * ylen), dpi=300)
        ims = np.zeros_like(axes)

        # Plot lens light
        axes[0, 0].set_title("Parametric model")
        axes[0, 1].set_title("Reference model at I0")
        axes[0, 2].set_title("Spectral index")
        ims[0, 0] = axes[0, 0].imshow(slp, **rendering)
        ims[0, 1] = axes[0, 1].imshow(sln, **rendering)
        ims[0, 2] = axes[0, 2].imshow(sla, **rendering)

        axes = axes.flatten()
        ims = ims.flatten()
        for ii, (energy_range, fld) in enumerate(zip(grid.spectral.color_ranges, sl)):
            energy, energy_unit = energy_range.center.value, energy_range.center.unit
            ii += 3

            axes[ii].set_title(f"{energy:.4f} {energy_unit}")
            ims[ii] = axes[ii].imshow(
                fld, norm=plotting_config.norm, vmin=vmin, vmax=vmax, **rendering
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
