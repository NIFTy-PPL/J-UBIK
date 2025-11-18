from dataclasses import dataclass, field
from os import makedirs
from os.path import join
from typing import Union

from charm_lensing.physical_models import HybridModel
from charm_lensing.physical_models.multifrequency_models.vstack_model import VstackModel
import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np
from astropy import units as u

from ....grid import Grid
from .plotting_base import get_alpha_and_reference, get_position_or_samples_of_model
from ..parse.plotting import FieldPlottingConfig, MultiFrequencyPlottingConfig


def _get_light(light: HybridModel | VstackModel):
    if isinstance(light, HybridModel):
        return light.parametric, *get_alpha_and_reference(light)

    elif isinstance(light, VstackModel):
        _jwst_alpha, _jwst_reference = get_alpha_and_reference(light.infrared)
        _alma_alpha, _alma_reference = get_alpha_and_reference(light.microwave)

        def jwst_parametric(x):
            return light.infrared.parametric(light.outer_2_inner(x, "infrared"))

        def jwst_alpha(x):
            if _jwst_alpha is None:
                return np.zeros((2, 2))
            else:
                return _jwst_alpha(light.outer_2_inner(x, "infrared"))

        def jwst_reference(x):
            if _jwst_reference is None:
                return np.zeros((2, 2))
            else:
                return _jwst_reference(light.outer_2_inner(x, "infrared"))

        def alma_parametric(x):
            return light.microwave.parametric(light.outer_2_inner(x, "microwave"))

        def alma_alpha(x):
            if _alma_alpha is None:
                return np.zeros((2, 2))
            else:
                return _alma_alpha(light.outer_2_inner(x, "infrared"))

        def alma_reference(x):
            if _alma_reference is None:
                return np.zeros((2, 2))
            else:
                return _alma_reference(light.outer_2_inner(x, "infrared"))

        return (
            alma_parametric,
            alma_alpha,
            alma_reference,
            jwst_parametric,
            jwst_alpha,
            jwst_reference,
        )

    else:
        ValueError("Only HybridModel or VstackModel")


def _plot_meta_parameters(ims, axes, samples, models, plotting_config):
    reference_plotting_config: FieldPlottingConfig = plotting_config.reference
    spectral_index_plotting_config: FieldPlottingConfig = plotting_config.alpha
    filter_plotting_config = plotting_config.combined
    rendering = filter_plotting_config.rendering

    para, alpha, ref = [
        get_position_or_samples_of_model(samples, m, True) for m in models[:3]
    ]

    if len(para.shape) > 2:
        para = para.mean(axis=0)

    # Plot lens light
    axes[0].set_title("Parametric model")
    axes[1].set_title("Reference model at I0")
    axes[2].set_title("Spectral index")
    ims[0] = axes[0].imshow(para, **rendering)
    ims[1] = axes[1].imshow(
        ref,
        vmin=reference_plotting_config.get_min(ref),
        vmax=reference_plotting_config.get_max(ref),
        norm=reference_plotting_config.norm,
        **rendering,
    )
    ims[2] = axes[2].imshow(
        alpha,
        vmin=spectral_index_plotting_config.get_min(alpha),
        vmax=spectral_index_plotting_config.get_max(alpha),
        norm=spectral_index_plotting_config.norm,
        **rendering,
    )

    if len(models) > 3:
        para, alpha, ref = [
            get_position_or_samples_of_model(samples, m, True) for m in models[3:]
        ]

        if len(para.shape) > 2:
            para = para.mean(axis=0)

        # Plot lens light
        axes[3].set_title("(jwst) Parametric model")
        axes[4].set_title("(jwst) Reference model at I0")
        axes[5].set_title("(jwst) Spectral index")
        ims[3] = axes[3].imshow(para, **rendering)
        ims[4] = axes[4].imshow(
            ref,
            vmin=reference_plotting_config.get_min(ref),
            vmax=reference_plotting_config.get_max(ref),
            norm=reference_plotting_config.norm,
            **rendering,
        )
        ims[5] = axes[5].imshow(
            alpha,
            vmin=spectral_index_plotting_config.get_min(alpha),
            vmax=spectral_index_plotting_config.get_max(alpha),
            norm=spectral_index_plotting_config.norm,
            **rendering,
        )


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

    source_light = lens_system.source_plane_model.light_model
    meta_models = list(_get_light(source_light))

    freq_len = len(grid.spectral)
    xlen = 3 if (freq_len < 4 or not isinstance(source_light, VstackModel)) else 6
    ylen = 1 + int(np.ceil(freq_len / xlen))

    for ii, model in enumerate(meta_models):
        if model is None:
            meta_models[ii] = lambda _: np.zeros((2, 2))

    filter_plotting_config = plotting_config.combined
    rendering = filter_plotting_config.rendering
    rendering["extent"] = lens_system.source_plane_model.space.extend().extent

    def plot_source(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        print("Plotting source light")

        sl = get_position_or_samples_of_model(position_or_samples, source_light, True)
        vmin, vmax = filter_plotting_config.get_min(sl), None

        fig, axes = plt.subplots(ylen, xlen, figsize=(3 * xlen, 3 * ylen), dpi=300)
        ims = np.zeros_like(axes)

        _plot_meta_parameters(
            ims[0], axes[0], position_or_samples, meta_models, plotting_config
        )

        axes = axes.flatten()
        ims = ims.flatten()
        for ii, (energy_range, fld) in enumerate(zip(grid.spectral, sl)):
            energy = energy_range.center.to(u.um, equivalencies=u.spectral())
            ii += xlen

            axes[ii].set_title(f"{energy.value:.4f} {energy.unit}")
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
