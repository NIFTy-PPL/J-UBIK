from os import makedirs
from os.path import join
from typing import Union

import matplotlib.pyplot as plt
import nifty.re as jft

from .plotting_base import get_position_or_samples_of_model


def build_color_components_plotting(
    sky_model,
    results_directory: str | None,
    substring="",
):
    from charm_lensing.physical_models.multifrequency_models.colormix_model import (
        ColorMix,
    )

    if not isinstance(sky_model, ColorMix):
        return lambda pos, state=None: None

    if results_directory is not None:
        colors_directory = join(
            results_directory, f"colors_{substring}" if substring != "" else "colors"
        )
        makedirs(colors_directory, exist_ok=True)
    N_comps = len(sky_model.components.components)

    def color_plot(
        position_or_samples: Union[dict, jft.Samples],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        mat_mean, mat_std = get_position_or_samples_of_model(
            position_or_samples, sky_model.color_matrix
        )
        print()
        print("Color Mixing Matrix")
        print(mat_mean, "\n+-\n", mat_std)
        print()

        components_mean, _ = get_position_or_samples_of_model(
            position_or_samples, sky_model.components
        )
        mixed_mean, _ = get_position_or_samples_of_model(
            position_or_samples, sky_model.mixed_components
        )
        correlated_mean, _ = get_position_or_samples_of_model(
            position_or_samples, sky_model
        )

        fig, axes = plt.subplots(3, N_comps, figsize=(4 * N_comps, 9))
        for ax, cor, comp, mix in zip(
            axes.T, correlated_mean, components_mean, mixed_mean
        ):
            im0 = ax[0].imshow(cor, origin="lower", norm="log", interpolation="None")
            im1 = ax[1].imshow(comp, origin="lower", interpolation="None")
            im2 = ax[2].imshow(mix - comp, origin="lower", interpolation="None")
            plt.colorbar(im0, ax=ax[0])
            plt.colorbar(im1, ax=ax[1])
            plt.colorbar(im2, ax=ax[2])
            ax[0].set_title("corr=exp(Mixed)")
            ax[1].set_title("Components")
            ax[2].set_title("Mixed-Components")

        plt.tight_layout()
        if isinstance(state_or_none, jft.OptimizeVIState):
            plt.savefig(join(colors_directory, f"componets_{state_or_none.nit}.png"))
            plt.close()
        else:
            plt.show()

    return color_plot
