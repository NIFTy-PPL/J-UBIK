from dataclasses import asdict
from os import makedirs
from os.path import join
from typing import Any, Union

import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np
from nifty8.re.library.mf_model import CorrelatedMultiFrequencySky

from ....grid import Grid
from ..parse.plotting import LensSystemPlottingConfig
from .plotting_base import (
    FieldPlottingConfig,
    get_alpha_and_reference,
    get_position_or_samples_of_model,
)


def build_plot_lens_system(
    results_directory: str | None,
    plotting_config: LensSystemPlottingConfig,
    lens_system,
    grid: Grid,
    parametric_lens: bool = False,
    parametric_source: bool = False,
):
    from charm_lensing.lens_system import LensSystem
    from charm_lensing.physical_models.hybrid_model import HybridModel
    from charm_lensing.physical_models.mean_model import MeanModel

    lens_system: LensSystem = lens_system

    if results_directory is not None:
        lens_dir = join(results_directory, "lens")
        makedirs(lens_dir, exist_ok=True)

    tshape = lens_system.get_forward_model_parametric().target.shape
    # FIXME: This should be handled by a source with shape 3
    xlen = tshape[0] + 2 if len(tshape) == 3 else 3

    lens_light = lens_system.lens_plane_model.light_model
    lens_light_alpha, lens_light_reference = get_alpha_and_reference(
        lens_system.lens_plane_model.light_model
    )

    if parametric_lens:
        lensed_light = lens_system.get_forward_model_parametric(only_source=True)
        if isinstance(lens_system.lens_plane_model.convergence_model, MeanModel):
            convergence = lens_system.lens_plane_model.convergence_model
        if isinstance(lens_system.lens_plane_model.convergence_model, HybridModel):
            convergence = lens_system.lens_plane_model.convergence_model.parametric
        convergence_nonparametric = lambda _: np.zeros((12, 12))
    else:
        lensed_light = lens_system.get_forward_model_full(only_source=True)
        convergence = lens_system.lens_plane_model.convergence_model
        convergence_nonparametric = (
            lens_system.lens_plane_model.convergence_model.nonparametric
        )

    if parametric_source:
        lensed_light = lens_system.get_forward_model_parametric_source(
            parametric_lens=parametric_lens, only_source=True
        )
        if isinstance(lens_system.source_plane_model.light_model, MeanModel):
            source_light = lens_system.source_plane_model.light_model
        if isinstance(lens_system.source_plane_model.light_model, HybridModel):
            source_light = lens_system.source_plane_model.light_model.parametric
        source_light_alpha = source_light_reference = lambda _: np.zeros((12, 12))

    else:
        source_light = lens_system.source_plane_model.light_model
        source_light_alpha, source_light_reference = get_alpha_and_reference(
            lens_system.source_plane_model.light_model
        )

    lens_light_extent = lens_system.lens_plane_model.space.extent
    lens_mass_extent = lens_system.lens_plane_model.space.extend().extent
    source_extent = lens_system.source_plane_model.space.extend().extent

    rendering = dict(interpolation="None", origin="lower")

    models = [
        source_light,
        lens_light,
        lensed_light,
        convergence,
        convergence_nonparametric,
        lens_light_alpha,
        lens_light_reference,
        source_light_alpha,
        source_light_reference,
    ]

    for ii, model in enumerate(models):
        if model is None:
            models[ii] = lambda _: np.zeros((2, 2))

    def plot_lens_system(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        print("Plotting lens system")

        sl, ll, ldl, cc, ccn, lla, lln, sla, sln = [
            get_position_or_samples_of_model(position_or_samples, model, True)
            for model in models
        ]

        # FIXME: This should be handled by a source with shape 3
        if len(ldl.shape) == 2:
            ldl = ldl[None]
        if len(ll.shape) == 2:
            ll = ll[None]
        if len(sl.shape) == 2:
            sl = sl[None]

        fig, axes = plt.subplots(3, xlen, figsize=(3 * xlen, 8), dpi=300)
        ims = np.zeros_like(axes)
        light_offset = 2

        # Plot lens light
        axes[0, 0].set_title("Lens light alpha")
        axes[0, 1].set_title("Lens light reference")
        ims[0, 0] = axes[0, 0].imshow(lla, **rendering)
        ims[0, 1] = axes[0, 1].imshow(lln, **rendering)

        # Plot source light
        axes[1, 0].set_title("Source light alpha")
        axes[1, 1].set_title("Source light reference")
        ims[1, 0] = axes[1, 0].imshow(
            sla,
            extent=source_extent,
            **asdict(plotting_config.source.alpha),
            **rendering,
        )
        ims[1, 1] = axes[1, 1].imshow(
            sln,
            extent=source_extent,
            **asdict(plotting_config.source.reference),
            **rendering,
        )

        # Plot mass field
        axes[2, 0].set_title("Mass parmetric")
        axes[2, 1].set_title("Mass nonparametric")
        ims[2, 0] = axes[2, 0].imshow(
            cc,
            extent=lens_mass_extent,
            **asdict(plotting_config.lens_mass),
            **rendering,
        )
        ims[2, 1] = axes[2, 1].imshow(
            ccn,
            extent=lens_mass_extent,
            **asdict(plotting_config.lens_mass),
            **rendering,
        )

        if plotting_config.share_source_vmin_vmax:
            vmin_source = plotting_config.source.combined.get_min(sl)
            vmax_source = plotting_config.source.combined.get_max(sl)
            vmin_lensed_light = plotting_config.source.combined.get_min(sl)
            vmax_lensed_light = plotting_config.source.combined.get_max(sl)

        vmin_lens = plotting_config.lens_light.combined.get_min(ll)
        vmax_lens = plotting_config.lens_light.combined.get_max(ll)

        for ii, energy_range in enumerate(grid.spectral.color_ranges):
            energy, energy_unit = energy_range.center.value, energy_range.center.unit
            ename = f"{energy:.4f} {energy_unit}"
            axes[0, ii + light_offset].set_title(f"Lens light {ename}")
            ims[0, ii + light_offset] = axes[0, ii + light_offset].imshow(
                ll[ii],
                extent=lens_light_extent,
                norm=plotting_config.lens_light.combined.norm,
                vmin=vmin_lens,
                vmax=vmax_lens,
                **rendering,
            )

            if not plotting_config.share_source_vmin_vmax:
                vmin_source = plotting_config.source.combined.get_min(sl[ii])
                vmax_source = plotting_config.source.combined.get_max(sl[ii])
                vmin_lensed_light = plotting_config.source.combined.get_min(sl[ii])
                vmax_lensed_light = plotting_config.source.combined.get_max(sl[ii])

            axes[1, ii + light_offset].set_title(f"Source light {ename}")
            ims[1, ii + light_offset] = axes[1, ii + light_offset].imshow(
                sl[ii],
                extent=source_extent,
                norm=plotting_config.source.combined.norm,
                vmin=vmin_source,
                vmax=vmax_source,
                **rendering,
            )

            axes[2, ii + light_offset].set_title(f"Lensed light {ename}")
            ims[2, ii + light_offset] = axes[2, ii + light_offset].imshow(
                ldl[ii],
                extent=lens_light_extent,
                norm=plotting_config.source.combined.norm,
                vmin=vmin_lensed_light,
                vmax=vmax_lensed_light,
                **rendering,
            )

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)

        fig.tight_layout()

        if state_or_none is not None:
            fig.savefig(join(lens_dir, f"{state_or_none.nit:02d}.png"), dpi=300)
            plt.close()
        else:
            plt.show()

    return plot_lens_system
