import nifty8.re as jft

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np

from typing import Optional


def find_closest_factors(number):
    """
    Finds two integers whose multiplication is larger or equal to the input number,
    with the two output numbers being as close together as possible.

    Args:
        number: The input integer number.

    Returns:
        A tuple containing two integers (x, y) such that x * y >= number and the
        difference between x and y is minimized. If no such factors exist, returns
        None.
    """

    # Start with the square root of the number.
    ii = int(np.ceil(number**0.5))

    jj, kk = ii, ii

    jminus = kminus = 0
    while ((jj-jminus)*(kk-kminus) >= number):
        if kminus == jminus:
            jminus += 1
        else:
            kminus += 1

    if jminus == kminus:
        return jj-jminus, kk-kminus+1
    return jj-jminus+1, kk-kminus


def build_plot(
    data_dict: dict,
    sky_model_with_key: jft.Model,
    sky_model: jft.Model,
    small_sky_model: jft.Model,
    results_directory: str,
    plotting_config: dict,
    alpha: jft.Model,
):
    from jubik0.jwst.mock_data.mock_evaluation import redchi2
    from jubik0.jwst.mock_data.mock_plotting import display_text
    from os.path import join
    from os import makedirs

    residual_dir = join(results_directory, 'residuals')
    sky_dir = join(results_directory, 'sky')
    makedirs(residual_dir, exist_ok=True)
    makedirs(sky_dir, exist_ok=True)

    norm = plotting_config.get('norm', Normalize)
    sky_extent = plotting_config.get('sky_extent', None)
    plot_sky = plotting_config.get('plot_sky', True)

    def sky_plot_residuals(samples: jft.Samples, x: jft.OptimizeVIState):
        print(f"Results: {results_directory}")

        ylen = len(data_dict)
        fig, axes = plt.subplots(ylen, 4, figsize=(12, 3*ylen), dpi=300)
        ims = np.zeros_like(axes)
        for ii, (dkey, data) in enumerate(data_dict.items()):

            dm = data['data_model']
            dd = data['data']
            std = data['std']
            mask = data['mask']
            cm = data['correction_model']

            model_data = []
            for si in samples:
                tmp = np.zeros_like(dd)
                val = sky_model_with_key(si)
                if cm is not None:
                    while isinstance(si, jft.Vector):
                        si = si.tree
                    val = val | si
                tmp[mask] = dm(val)
                model_data.append(tmp)

            if cm is not None:
                sh_m, sh_s = jft.mean_and_std(
                    [cm.shift_prior(s) for s in samples])
                ro_m, ro_s = jft.mean_and_std(
                    [cm.rotation_prior(s) for s in samples])
                ro_m, ro_s = ro_m[0], ro_s[0]
                sh_m, sh_s = (sh_m.reshape(2), sh_s.reshape(2))
            else:
                sh_m, sh_s, ro_m, ro_s = (0, 0), (0, 0), 0, 0

            model_mean = jft.mean(model_data)
            redchi_mean, redchi2_std = jft.mean_and_std(
                [redchi2(dd[mask], m[mask], std[mask], dd[mask].size) for m in model_data])

            max_d, min_d = np.nanmax(dd), np.nanmin(dd)
            min_d = np.max((min_d, 0.1))
            axes[ii, 1].set_title(f'Data {dkey}')
            ims[ii, 1] = axes[ii, 1].imshow(
                dd, origin='lower', norm=norm(vmin=min_d, vmax=max_d))
            axes[ii, 2].set_title(
                f'Data model ({sh_m[0]:.1e}+-{sh_s[0]:.1e}, {sh_m[1]:.1e}+-{sh_s[1]:.1e},\n{ro_m:.1e}+-{ro_s:.1e})')
            ims[ii, 2] = axes[ii, 2].imshow(
                model_mean, origin='lower', norm=norm(vmin=min_d, vmax=max_d))
            axes[ii, 3].set_title('Data - Data model')
            ims[ii, 3] = axes[ii, 3].imshow(
                (dd - model_mean)/std, origin='lower', vmin=-3, vmax=3, cmap='RdBu_r')

            chi = '\n'.join((
                f'redChi2: {redchi_mean:.2f} +/- {redchi2_std:.2f}',
            ))

            display_text(axes[ii, 3], chi)

        # Alpha field
        alpha_mean = jft.mean([alpha(si) for si in samples])
        axes[0, 0].set_title('Alpha field')
        ims[0, 0] = axes[0, 0].imshow(alpha_mean, origin='lower')

        # Calculate sky
        mean_small = jft.mean([small_sky_model(si) for si in samples])
        ma, mi = jft.max(mean_small), jft.min(mean_small)

        for ii, filter_name in enumerate(sky_model_with_key.target.keys()):
            axes[ii+1, 0].set_title(f'Sky {filter_name}')
            ims[ii+1, 0] = axes[ii+1, 0].imshow(
                mean_small[ii], origin='lower', norm=norm(vmax=ma, vmin=mi))

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(residual_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    def plot_sky_with_samples(samples: jft.Samples, x: jft.OptimizeVIState):
        ylen, xlen = find_closest_factors(len(samples)+4)

        samps_big = [sky_model(si) for si in samples]
        mean, std = jft.mean_and_std(samps_big)
        mean_small, std_small = jft.mean_and_std(
            [small_sky_model(si) for si in samples])
        flds = [mean_small, std_small/mean_small, mean, std/mean] + samps_big

        for ii, filter_name in enumerate(sky_model_with_key.target.keys()):
            fig, axes = plt.subplots(
                ylen, xlen, figsize=(2*xlen, 1.5*ylen), dpi=300)
            for ax, fld in zip(axes.flatten(), flds):
                im = ax.imshow(
                    fld[ii], origin='lower', norm=norm(), extent=sky_extent)
                fig.colorbar(im, ax=ax, shrink=0.7)
            fig.tight_layout()
            fig.savefig(
                join(sky_dir, f'{x.nit:02d}_{filter_name}.png'), dpi=300)
            plt.close()

    def sky_plot(samples: jft.Samples, x: jft.OptimizeVIState):
        print(f'Plotting: {x.nit}')
        sky_plot_residuals(samples, x)
        if plot_sky:
            plot_sky_with_samples(samples, x)

    return sky_plot


def plot_sky(sky, data_dict, norm=LogNorm):

    ylen = len(data_dict)
    fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
    ims = []
    for ii, (dkey, data) in enumerate(data_dict.items()):

        dm = data['data_model']
        dd = data['data']
        std = data['std']
        mask = data['mask']

        model_data = np.zeros_like(dd)
        model_data[mask] = dm(sky)

        axes[ii, 0].set_title(f'Data {dkey}')
        ims.append(axes[ii, 0].imshow(dd, origin='lower', norm=norm()))
        axes[ii, 1].set_title('Data model')
        ims.append(axes[ii, 1].imshow(
            model_data, origin='lower', norm=norm()))
        axes[ii, 2].set_title('Data - Data model')
        ims.append(axes[ii, 2].imshow((dd - model_data)/std,
                   origin='lower', vmin=-3, vmax=3, cmap='RdBu_r'))

    for ax, im in zip(axes.flatten(), ims):
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.show()
