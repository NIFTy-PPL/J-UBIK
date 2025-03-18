from ....parse.instruments.resolve.plotting.standard_plotting import (
    PLOTTING_KWARGS_DEFAULT, PlottingKwargs)

import nifty8.re as jft
from nifty8.logger import logger

import matplotlib.pyplot as plt
import numpy as np

from os import makedirs


def build_standard_plot(
    sky: jft.Model,
    output_directory: str,
    plotting_kwargs: PlottingKwargs = PLOTTING_KWARGS_DEFAULT,
):
    logger.info(f'Output: {output_directory}')
    makedirs(output_directory, exist_ok=True)

    def callback(samples, state):
        logger.info(f'Plotting iteration {state.nit} in: {output_directory}')

        try:
            sky_mean = jft.mean([sky(x) for x in samples])
        except ZeroDivisionError:
            # NOTE : MAP solution
            sky_mean = sky(samples.pos)

        pols, ts, freqs, *_ = sky_mean.shape
        fig, axes = plt.subplots(pols, freqs, figsize=(freqs*4, pols*3))

        vmin = max(plotting_kwargs.vmin, sky_mean.min())
        vmax = min(plotting_kwargs.vmax, sky_mean.max())
        settings = dict(vmin=vmin, vmax=vmax, norm='log', origin='lower')

        if freqs == 1:
            if pols == 1:
                axes = [axes]

            for poli, ax in enumerate(axes):
                f = sky_mean[poli, 0, 0].T
                if poli > 0:
                    f = np.abs(f)

                im = ax.imshow(f, **settings)
                plt.colorbar(im, ax=ax)

        elif pols == 1:
            if freqs == 1:
                axes = [axes]

            for freqi, ax in enumerate(axes):
                im = ax.imshow(
                    sky_mean[0, 0, freqi].T, **settings)
                plt.colorbar(im, ax=ax)

        else:
            for poli, pol_axes in enumerate(axes):
                for freqi, ax in enumerate(pol_axes):
                    if poli > 0:
                        f = np.abs(f)
                    f = sky_mean[poli, 0, freqi].T
                    im = ax.imshow(f, **settings)
                    plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(f"{output_directory}/resolve_iteration_{state.nit}.png")
        plt.close()

    return callback
