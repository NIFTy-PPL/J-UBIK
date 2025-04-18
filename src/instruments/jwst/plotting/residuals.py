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
    _determine_xpos,
    _determine_ypos,
    _get_model_samples_or_position,
    determine_xlen_residuals,
    _get_data_model_and_chi2,
    plot_data_data_model_residuals,
    get_shift_rotation_correction,
)


def build_plot_sky_residuals(
    results_directory: str,
    filter_projector: FilterProjector,
    data_dict: dict,
    sky_model_with_key: jft.Model,
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

    if isinstance(sky_model_with_key.target, dict):
        ylen = len(next(iter(sky_model_with_key.target)))
    else:
        ylen = len(sky_model_with_key.target)
    xlen = 3 + determine_xlen_residuals(data_dict, xmax_residuals)

    rendering = plotting_config.sky.rendering

    def sky_residuals(
        position_or_samples: Union[dict, jft.Samples],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        print(f"Results: {results_directory}")
        print("Plotting residuals")

        fig, axes = plt.subplots(ylen, xlen, figsize=(3 * xlen, 3 * ylen), dpi=300)
        ims = np.zeros_like(axes)
        if ylen == 1:
            ims = ims[None]
            axes = axes[None]

        sky_or_skies = _get_model_samples_or_position(
            position_or_samples, sky_model_with_key
        )

        if isinstance(position_or_samples, jft.Samples):
            sky = jft.mean(sky_or_skies)
        else:
            sky = sky_or_skies
        sky_max, sky_min = (
            plotting_config.sky.get_max(np.max(list(tree.map(np.max, sky).values()))),
            plotting_config.sky.get_min(np.min(list(tree.map(np.min, sky).values()))),
        )

        for skey, ypos in filter_projector.keys_and_index.items():
            axes[ypos, 0].set_title(f"Sky {skey}")
            ims[ypos, 0] = axes[ypos, 0].imshow(
                sky[skey],
                norm=plotting_config.sky.norm,
                vmax=sky_max,
                vmin=sky_min,
                **rendering,
            )

        for dkey, data in data_dict.items():
            xpos_residual = _determine_xpos(dkey)
            ypos = _determine_ypos(dkey, filter_projector, plotting_config.ylen_offset)
            if xpos_residual > xlen - 1:
                continue

            data_model = data["data_model"]
            data_i = data["data"]
            std = data["std"]
            mask = data["mask"]

            model_mean, (redchi_mean, redchi_std) = _get_data_model_and_chi2(
                position_or_samples,
                sky_or_skies,
                data_model=data_model,
                data=data_i,
                mask=mask,
                std=std,
            )

            if xpos_residual == 3:
                ims[ypos, 1:] = plot_data_data_model_residuals(
                    ims[ypos, 1:],
                    axes[ypos, 1:],
                    data_key=dkey,
                    data=data_i,
                    data_model=model_mean,
                    std=std if std_relative else 1.0,
                    plotting_config=residual_plotting_config,
                )

                if display_pointing:
                    (sh_m, sh_s), (ro_m, ro_s) = get_shift_rotation_correction(
                        position_or_samples,
                        data_model.rotation_and_shift.correction_model,
                    )
                    data_model_text = "\n".join(
                        (
                            f"dx={sh_m[0]:.1e}+-{sh_s[0]:.1e}",
                            f"dy={sh_m[1]:.1e}+-{sh_s[1]:.1e}",
                            f"dth={ro_m:.1e}+-{ro_s:.1e}",
                        )
                    )
                    display_text(axes[ypos, 2], data_model_text)

            else:
                ims[ypos, xpos_residual] = axes[ypos, xpos_residual].imshow(
                    (data_i - model_mean) / std,
                    vmin=-3,
                    vmax=3,
                    cmap="RdBu_r",
                    **rendering,
                )

            if display_chi2:
                chi = "\n".join((f"redChi2: {redchi_mean:.2f} +/- {redchi_std:.2f}",))
                display_text(axes[ypos, xpos_residual], chi)

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
