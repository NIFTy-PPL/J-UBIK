import numpy as np
import nifty8.re as jft
import matplotlib.pyplot as plt

from os.path import join
from os import makedirs

from .mock_evaluation import (
    wasserstein_distance, source_distortion_ratio, redchi2, wmse,
    smallest_enclosing_mask, get_power)


def display_text(ax: plt.Axes, text: dict, **kwargs):
    '''Display text on plot
    ax: matplotlib axis
    text: dict or str (default: {'s': str, 'color': 'white'})
    kwargs:
    - keyword: str
        options: 'top_left' (default), 'top_right', 'bottom_left', 'bottom_right'
    - x_offset_ticker: float (default: 0)
    - y_offset_ticker: float (default: 0)
    '''
    keyword = kwargs.get('keyword', 'top_left')
    x_offset_ticker = kwargs.get('x_offset_ticker', 0)
    y_offset_ticker = kwargs.get('y_offset_ticker', 0)

    if type(text) is str:
        text = dict(
            s=text,
            color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'),
        )

    if keyword == 'top_left':
        ax.text(x=0.05 + x_offset_ticker*0.05,
                y=0.95 - y_offset_ticker*0.05,
                ha='left',
                va='top',
                transform=ax.transAxes,
                **text)
    elif keyword == 'top_right':
        ax.text(x=0.95 - x_offset_ticker*0.05,
                y=0.95 - y_offset_ticker*0.05,
                ha='right',
                va='top',
                transform=ax.transAxes,
                **text)
    elif keyword == 'bottom_left':
        ax.text(x=0.05 + x_offset_ticker*0.05,
                y=0.05 + y_offset_ticker*0.05,
                ha='left',
                va='bottom',
                transform=ax.transAxes,
                **text)
    elif keyword == 'bottom_right':
        ax.text(x=0.95 - x_offset_ticker*0.05,
                y=0.05 + y_offset_ticker*0.05,
                ha='right',
                va='bottom',
                transform=ax.transAxes,
                **text)
    else:
        raise ValueError(
            "Invalid keyword. Use 'top_left', 'top_right', 'bottom_left', or 'bottom_right'.")


def build_mock_plot(
    data_set, comparison_sky, internal_sky_key, sky_model, res_dir, eval_mask
):
    datas = [ll['data'] for ll in data_set.values()]
    data_models = [ll['data_model'] for ll in data_set.values()]
    masks = [ll['mask'] for ll in data_set.values()]
    stds = [ll['std'] for ll in data_set.values()]
    corrs = [ll['correction_model'] for ll in data_set.values()]

    out_dir = join(res_dir, 'residuals')
    makedirs(out_dir, exist_ok=True)

    eval_comp_sky = np.zeros_like(comparison_sky)
    eval_self_sky = np.zeros_like(comparison_sky)

    ii, smallest_mask = smallest_enclosing_mask(eval_mask)
    reshape = eval_mask.shape[0] - 2*ii
    true_power_spectrum = get_power(comparison_sky, smallest_mask, reshape)

    def plot_pspec(samples, x):
        YLIMS = (1e2, 1e11)

        skys = [sky_model(si) for si in samples]

        pws = [get_power(sky, smallest_mask, reshape) for sky in skys]
        pw = get_power(jft.mean(skys), smallest_mask, reshape)

        fig, ax = plt.subplots(1, 1, figsize=(9, 3), dpi=300)
        ax.plot(pw, label='reco', color='blue', linewidth=0.5)
        for pw in pws:
            ax.plot(pw, color='blue', alpha=0.5, linewidth=0.3)
        ax.plot(true_power_spectrum, label='true',
                color='orange', linewidth=0.5)
        plt.ylim(YLIMS)
        ax.loglog()
        ax.legend()
        fig.savefig(join(out_dir, f'pspec_{x.nit:02d}.png'), dpi=300)
        plt.close()

    def sky_plot(samples, x):
        sky = jft.mean([sky_model(si) for si in samples])

        eval_comp_sky[eval_mask] = comparison_sky[eval_mask]
        eval_self_sky[eval_mask] = sky[eval_mask]

        vals = dict(
            sdr=source_distortion_ratio(eval_comp_sky, eval_self_sky),
            wd=wasserstein_distance(eval_comp_sky.reshape(-1),
                                    eval_self_sky.reshape(-1)),
            # cc=cross_correlation(eval_comp_sky, eval_self_sky),
        )

        ylen = 1+len(datas)
        fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
        ims = []
        for ii, (d, std, dm, corr, mask) in enumerate(
                zip(datas, stds, data_models, corrs, masks)):
            model_data = []
            for si in samples:
                tmp = np.zeros_like(d)
                val = {internal_sky_key: sky_model(si)}
                while isinstance(si, jft.Vector):
                    si = si.tree
                val = val | si
                tmp[mask] = dm(val)
                model_data.append(tmp)

            mod_mean = jft.mean(model_data)
            redchi_mean, redchi2_std = jft.mean_and_std(
                [redchi2(d, m, std, d.size) for m in model_data])

            axes[ii, 0].set_title('Data')
            ims.append(axes[ii, 0].imshow(d, origin='lower'))
            axes[ii, 1].set_title('Data model')
            ims.append(axes[ii, 1].imshow(mod_mean, origin='lower'))
            axes[ii, 2].set_title('Data residual')
            ims.append(axes[ii, 2].imshow((d - mod_mean)/std,
                       origin='lower', vmin=-3, vmax=3, cmap='RdBu_r'))
            chi = '\n'.join((
                f'MSE/var: {wmse(d, mod_mean, std):.2f}',
                f'redChi2: {redchi_mean:.2f} +/- {redchi2_std:.2f}',
            ))

            display_text(axes[ii, 2], chi)

        axes[ii+1, 0].set_title('Sky')
        ims.append(axes[ii+1, 0].imshow(comparison_sky, origin='lower'))
        axes[ii+1, 1].set_title('Sky model')
        ims.append(axes[ii+1, 1].imshow(sky, origin='lower'))
        axes[ii+1, 2].set_title('Sky residual')
        ims.append(axes[ii+1, 2].imshow(
            (comparison_sky - sky)/comparison_sky, origin='lower',
            vmin=-0.3, vmax=0.3, cmap='RdBu_r'))

        axes[ii+1, 0].contour(eval_mask, levels=1, colors='orange')
        axes[ii+1, 1].contour(eval_mask, levels=1, colors='orange')
        axes[ii+1, 2].contour(eval_mask, levels=1, colors='orange')

        ss = '\n'.join(
            [f'{k}: {v:.3f}' if k != 'cc' else f'{k}: {v:e}' for k, v in vals.items()])
        display_text(axes[ii+1, 2], ss)
        for ax, im in zip(axes.flatten(), ims):
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(out_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    def report_correction(samples, x):
        print('data', 'true-reported', 'corr')
        for key, val in data_set.items():
            shift_true, shift_reported = val['shift']['true'], val['shift']['reported']

            cm = val['correction_model']
            if cm is not None:
                corr, cors = jft.mean_and_std(
                    [cm.prior_model(s) for s in samples])
                corr, cors = (corr.reshape(2), cors.reshape(2))
                cor = f'[{corr[0]:.1f}+-{cors[0]:.1f}, {corr[1]:.1f}+-{cors[1]:.1f}]'
            else:
                cor = '[0+-0, 0+-0]'
            print(key, np.array(shift_true) - np.array(shift_reported), cor)

    def plot(samples, x):
        print(f'Results: {res_dir}')
        sky_plot(samples, x)

        try:
            # plot_pspec(samples, x)
            pass
        except:
            None

    return plot


def sky_model_check(
    key,
    sky_model,
    comp_sky,


):
    m = sky_model(jft.random_like(key, sky_model.domain))

    fig, axis = plt.subplots(1, 3)
    im0 = axis[0].imshow(comp_sky, origin='lower')
    im1 = axis[1].imshow(m, origin='lower')
    im2 = axis[2].imshow(comp_sky-m, origin='lower', cmap='RdBu_r')
    axis[0].set_title('sky')
    axis[1].set_title('model')
    axis[2].set_title('residual')
    for im, ax in zip([im0, im1, im2], axis):
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.show()
