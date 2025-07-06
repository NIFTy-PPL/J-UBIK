from dataclasses import asdict
from os import makedirs
from os.path import join
from typing import Any, Union, Callable
from collections import namedtuple

import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np

from ....grid import Grid
from ..parse.plotting import LensSystemPlottingConfig
from .plotting_base import (
    FieldPlottingConfig,
    get_alpha_and_reference,
    get_position_or_samples_of_model,
)
from ....sky_model.multifrequency.spectral_product_mf_sky import SpectralProductSky

LensSystemPlottingModels = namedtuple(
    "LensSystemPlottingModels",
    [
        "source_light",
        "lens_light",
        "lensed_light",
        "convergence",
        "convergence_nonparametric",
        "lens_light_alpha",
        "lens_light_reference",
        "source_light_alpha",
        "source_light_reference",
    ],
)


def _build_lens_plotting_models(
    lens_system,
    parametric_lens: bool = False,
    parametric_source: bool = False,
) -> LensSystemPlottingModels:
    from charm_lensing.lens_system import LensSystem
    from charm_lensing.physical_models.hybrid_model import HybridModel
    from charm_lensing.physical_models.mean_model import MeanModel

    lens_system: LensSystem = lens_system

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
        convergence_nonparametric = None
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
        source_light_alpha = source_light_reference = None

    else:
        source_light = lens_system.source_plane_model.light_model
        source_light_alpha, source_light_reference = get_alpha_and_reference(
            lens_system.source_plane_model.light_model
        )

    models_dict = {
        "source_light": source_light,
        "lens_light": lens_light,
        "lensed_light": lensed_light,
        "convergence": convergence,
        "convergence_nonparametric": convergence_nonparametric,
        "lens_light_alpha": lens_light_alpha,
        "lens_light_reference": lens_light_reference,
        "source_light_alpha": source_light_alpha,
        "source_light_reference": source_light_reference,
    }

    # Adjusting the loop to work with the dictionary
    for key, model in models_dict.items():
        if model is None:
            models_dict[key] = lambda _: np.zeros((2, 2))

    return LensSystemPlottingModels(**models_dict)


def _build_all_skies(
    lens_system,
    lens_dir: str,
    grid: Grid,
    models: LensSystemPlottingModels,
    plotting_config: LensSystemPlottingConfig,
) -> Callable[[jft.Samples, jft.OptimizeVIState | None], None]:
    tshape = models.lensed_light.target.shape
    # FIXME: This should be handled by a source with shape 3
    xlen = tshape[0] + 2 if len(tshape) == 3 else 3
    lens_light_extent = lens_system.lens_plane_model.space.extent
    lens_mass_extent = lens_system.lens_plane_model.space.extend().extent
    source_extent = lens_system.source_plane_model.space.extend().extent
    rendering = dict(interpolation="None", origin="lower")

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
        axes[2, 0].set_title("Mass Full")
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
                try:
                    fig.colorbar(im, ax=ax, shrink=0.7)
                except:
                    continue

        fig.tight_layout()

        if state_or_none is not None:
            fig.savefig(join(lens_dir, f"{state_or_none.nit:02d}.png"), dpi=300)
            plt.close()
        else:
            plt.show()

    return plot_lens_system


def _build_models_for_rgb_plotting(
    lens_system,
    grid: Grid,
    parametric_lens: bool = False,
    parametric_source: bool = False,
) -> LensSystemPlottingModels:
    from charm_lensing.physical_models.hybrid_model import HybridModel
    from charm_lensing.lens_system import LensSystem
    import jax.numpy as jnp

    lens_system: LensSystem = lens_system
    source_light, lens_light = (
        lens_system.source_plane_model.light_model,
        lens_system.lens_plane_model.light_model,
    )

    target_shape = lens_light.target.shape
    assert _check_target_shape_for_rgb_plotting(target_shape), (
        f"Check target shape for model: {lens_light}"
    )

    msg_model = "For RGB plotting {type} needs to be an hybrid model"
    assert isinstance(source_light, HybridModel), msg_model.format(type="source light")
    assert isinstance(lens_light, HybridModel), msg_model.format(type="lens light")

    msg_model = "For RGB plotting {type} needs to be a SpectralProductSky"
    assert isinstance(source_light.nonparametric, SpectralProductSky), msg_model.format(
        type="source light"
    )
    assert isinstance(lens_light.nonparametric, SpectralProductSky), msg_model.format(
        type="lens light"
    )

    # Get the log frequencies
    freqs_min = np.log(grid.spectral[0].start.value)
    freqs_max = np.log(grid.spectral[-1].end.value)
    dist = (freqs_max - freqs_min) / 3
    binbounds = [freqs_min + dist * ii for ii in range(4)]
    logfreqs = np.array([(binbounds[ii + 1] + binbounds[ii]) / 2 for ii in range(3)])

    # setup source model
    source_spectral: SpectralProductSky = (
        lens_system.source_plane_model.light_model.nonparametric
    )
    ref_index, *_ = np.where(
        source_spectral.log_spectral_behavior.relative_log_frequencies.flatten() == 0
    )
    logreffreq_source = np.log(grid.spectral[ref_index[0]].center.value)
    perturbation_source = jft.Model(
        lambda x: jnp.array(
            [
                source_spectral.get_spectral_distribution_at_relative_log_frequency(
                    x, freq
                )
                for freq in (logfreqs - logreffreq_source)
            ]
        ),
        domain=source_spectral.domain,
    )
    source_plane_model = lens_system.source_plane_model.set_model(
        "light_model",
        lens_system.source_plane_model.light_model.set_model(
            "perturbation", perturbation_source
        ),
    )
    from charm_lensing.spaces import Space

    source_space = source_plane_model.space
    source_space_new = Space(
        shape=source_space.shape,
        distances=source_space.distances,
        space_key=source_space.space_key,
        extend_factor=source_space.extend_factor,
    )
    source_space_new.e_bin = logfreqs
    source_plane_model = source_plane_model.set_model("space", source_space_new)

    # setup lens light model
    lens_spectral: SpectralProductSky = (
        lens_system.lens_plane_model.light_model.nonparametric
    )
    ref_index, *_ = np.where(
        lens_spectral.log_spectral_behavior.relative_log_frequencies.flatten() == 0
    )
    logreffreq_lens = np.log(grid.spectral[ref_index[0]].center.value)
    perturbation_lens = jft.Model(
        lambda x: jnp.array(
            [
                lens_spectral.get_spectral_distribution_at_relative_log_frequency(
                    x, freq
                )
                for freq in (logfreqs - logreffreq_lens)
            ]
        ),
        domain=lens_spectral.domain,
    )

    lens_plane_model = lens_system.lens_plane_model.set_model(
        "light_model",
        lens_system.lens_plane_model.light_model.set_model(
            "perturbation", perturbation_lens
        ),
    )

    new_lens_system = LensSystem(
        lens_plane_model=lens_plane_model,
        source_plane_model=source_plane_model,
    )

    return _build_lens_plotting_models(
        new_lens_system,
        parametric_lens=parametric_lens,
        parametric_source=parametric_source,
    )


def _build_rgb_skies(
    lens_system,
    lens_dir: str,
    models: LensSystemPlottingModels,
    grid: Grid,
    plotting_config: LensSystemPlottingConfig,
) -> Callable[[jft.Samples, jft.OptimizeVIState | None], None]:
    lens_light_extent = lens_system.lens_plane_model.space.extent
    lens_mass_extent = lens_system.lens_plane_model.space.extend().extent
    source_extent = lens_system.source_plane_model.space.extend().extent
    rendering = dict(interpolation="None", origin="lower")

    xlen = 3

    def plot_lens_system(
        position_or_samples: Union[jft.Samples, dict],
        state_or_none: jft.OptimizeVIState | None = None,
    ):
        print("Plotting lens system")

        sl, lal, ldl, cc, ccn, lla, lln, sla, sln = [
            get_position_or_samples_of_model(position_or_samples, model, True)
            for model in models
        ]

        bbb = sl.shape[0] // 3
        sl = np.array(
            [
                sl[0:bbb].mean(axis=0),
                sl[bbb : 2 * bbb].mean(axis=0),
                sl[2 * bbb : -1].mean(axis=0),
            ]
        )
        sl = np.log(sl + 1)
        sl = sl / sl.max()

        lal = np.array(
            [
                lal[0:bbb].mean(axis=0),
                lal[bbb : 2 * bbb].mean(axis=0),
                lal[2 * bbb : -1].mean(axis=0),
            ]
        )
        lal = np.log(lal + 1)
        lal = lal / lal.max()

        ldl = np.array(
            [
                ldl[0:bbb].mean(axis=0),
                ldl[bbb : 2 * bbb].mean(axis=0),
                ldl[2 * bbb : -1].mean(axis=0),
            ]
        )
        ldl = np.log(ldl + 1)
        ldl = ldl / ldl.max()

        fig, axes = plt.subplots(3, xlen, figsize=(3 * xlen, 8), dpi=300)
        ims = np.zeros_like(axes)

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
        axes[2, 0].set_title("Mass Full")
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

        axes[0, 2].set_title("Lens light")
        ims[0, 2] = axes[0, 2].imshow(
            # Move colour axis to the end: (H, W, 3) – format Matplotlib expects [[1]]
            np.moveaxis(lal, 0, -1),
            extent=lens_light_extent,
            **rendering,
        )

        axes[1, 2].set_title("Source light")
        ims[1, 2] = axes[1, 2].imshow(
            np.moveaxis(sl, 0, -1),
            extent=source_extent,
            **rendering,
        )

        axes[2, 2].set_title("Lensed light")
        ims[2, 2] = axes[2, 2].imshow(
            np.moveaxis(ldl, 0, -1),
            extent=lens_light_extent,
            **rendering,
        )

        for ax, im in zip(axes[:, :2].flatten(), ims[:, :2].flatten()):
            if not isinstance(im, int):
                try:
                    fig.colorbar(im, ax=ax, shrink=0.7)
                except:
                    continue
        fig.tight_layout()

        if state_or_none is not None:
            fig.savefig(join(lens_dir, f"{state_or_none.nit:02d}.png"), dpi=300)
            plt.close()
        else:
            plt.show()

    return plot_lens_system


def _check_target_shape_for_rgb_plotting(target_shape: tuple[int]):
    """Only get rgb plotting if the shape fits."""
    if len(target_shape) == 2:
        return False

    if target_shape[0] < 3:
        return False

    return True


def build_plot_lens_system(
    results_directory: str | None,
    plotting_config: LensSystemPlottingConfig,
    grid: Grid,
    lens_system,
    parametric_lens: bool = False,
    parametric_source: bool = False,
) -> Callable[[jft.Samples, jft.OptimizeVIState | None], None]:
    if results_directory is not None:
        lens_dir = join(results_directory, "lens")
        makedirs(lens_dir, exist_ok=True)

    if plotting_config.rgb_plotting:
        return _build_rgb_skies(
            lens_system=lens_system,
            lens_dir=lens_dir,
            grid=grid,
            # models=_build_models_for_rgb_plotting(
            #     lens_system=lens_system,
            #     grid=grid,
            #     parametric_lens=parametric_lens,
            #     parametric_source=parametric_source,
            # ),
            models=_build_lens_plotting_models(
                lens_system=lens_system,
                parametric_lens=parametric_lens,
                parametric_source=parametric_source,
            ),
            plotting_config=plotting_config,
        )

    else:
        return _build_all_skies(
            lens_system=lens_system,
            lens_dir=lens_dir,
            grid=grid,
            models=_build_lens_plotting_models(
                lens_system=lens_system,
                parametric_lens=parametric_lens,
                parametric_source=parametric_source,
            ),
            plotting_config=plotting_config,
        )
