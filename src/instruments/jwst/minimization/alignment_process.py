from os.path import join

import nifty.re as jft
from jax import clear_caches, random


# General stuff
from ....likelihood import get_n_constrained_dof
from ....minimization.minimization_from_samples import (
    KLSettings,
    minimization_from_initial_samples,
)
from ....minimization_parser import MinimizationParser
from ....utils import get_config

# Jwst stuff
from ..likelihood.alignment_likelihood import AlignmentLikelihoodProducts
from ..plotting.plotting import build_plot_alignment_residuals
from ..plotting.plotting_base import FieldPlottingConfig


def _print_some_results(plotting, samples_fixpointing) -> None:
    for pa in plotting:
        m = pa.model[0]
        try:
            mean, std = jft.mean_and_std(
                [
                    m.sky_model.location.shift_and_rotation.shift(x)
                    for x in samples_fixpointing
                ]
            )
            jft.logger.info(f"{pa.filter} {mean} {std}")

        except IndexError:
            x = samples_fixpointing.pos
            mean = m.sky_model.location.shift_and_rotation.shift(x)
            jft.logger.info(f"{pa.filter} {mean}")


def alignment_minimization_process(
    config_path: str,
    results_directory: str,
    alignment: AlignmentLikelihoodProducts,
):
    cfg_mini_fixpointing = get_config(config_path)["minimization_fixpointing"]

    samples_fixpointing = None
    for name, likelihood, plotting in zip(
        ["convolved", "psf"],
        [alignment.likelihood.likelihood_convolved, alignment.likelihood.likelihood],
        [alignment.plotting.convolved, alignment.plotting.psf],
    ):
        jft.logger.info(f"Fixpointing: {name}")

        plot_alignment_residuals = build_plot_alignment_residuals(
            join(results_directory, "fixpointing"),
            plotting,
            FieldPlottingConfig(vmin=1e-4, norm="log"),
            name_append=f"_{name}",
        )

        mini_parser_fixpointing = MinimizationParser(
            cfg_mini_fixpointing, get_n_constrained_dof(likelihood), verbose=False
        )
        kl_settings_fixpointing = KLSettings(
            random_key=random.PRNGKey(cfg_mini_fixpointing.get("key", 42)),
            outputdir=join(results_directory, "fixpointing", name),
            minimization=mini_parser_fixpointing,
            callback=plot_alignment_residuals,
            kl_jit=False,
            residual_jit=True,
            resume=cfg_mini_fixpointing.get("resume", False),
            n_total_iterations=cfg_mini_fixpointing["n_total_iterations"],
            resume_from_pickle_path=cfg_mini_fixpointing.get("resume_from_pickle_path"),
        )

        jft.logger.info("Fix pointing reconstruction")
        samples_fixpointing, state_imaging = minimization_from_initial_samples(
            likelihood,
            kl_settings_fixpointing,
            starting_samples=samples_fixpointing,
        )

        try:
            _print_some_results(plotting, samples_fixpointing)
        except:
            pass

        # Needed for the next runs, otherwise clocking up by jax...
        clear_caches()

    # plot_alignment_residuals = build_plot_alignment_residuals(
    #     join(results_directory, "fixpointing"),
    #     plotting,
    #     FieldPlottingConfig(vmin=1e-4, norm="log"),
    #     name_append=f"_{name}",
    #     interactive=True,
    # )
    # plot_alignment_residuals(samples_fixpointing, state_imaging)

    return samples_fixpointing
