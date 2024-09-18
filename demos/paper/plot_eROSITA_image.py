import matplotlib.pyplot as plt
import numpy as np
import jubik0 as ju
from os.path import join
import nifty8.re as jft

from matplotlib.colors import LogNorm
from matplotlib import patches
from matplotlib.ticker import LogLocator
def plot(array, pixel_measure=None, pixel_factor =1, output_file=None, logscale=False, title=None, colorbar=True,
         figsize=(8, 8), dpi=200, n_rows=None,
         share_x=True, share_y=True, fs=14, alpha=0.8,
             bbox_info= [(7, 4), 7,  30, 'black'], **kwargs):
    n_cols= int(np.ceil(array.shape[0]/n_rows))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, dpi=dpi, sharex=share_x,
                             sharey=share_y)

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]
    if 'cmap' in kwargs.keys():
        pltargs = {"origin": "lower", "cmap": kwargs['cmap'] , 'interpolation':'None'}
    else:
        pltargs = {"origin": "lower", "cmap": "viridis", 'interpolation': 'None'}

    for i in range(array.shape[0]):
        ax[i].tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        if array[i].ndim != 2:
            raise ValueError("All arrays to plot must be 2-dimensional!")

        if "vmin" in kwargs:
            vmin = kwargs["vmin"]
        else:
            vmin = None
        if "vmax" in kwargs:
            vmax = kwargs["vmax"]
        else:
            vmax = None

        if logscale:
            if vmin is not None and float(vmin) == 0.:
                vmin = 1e-18
            pltargs["norm"] = LogNorm(vmin, vmax)
        else:
            kwargs.update({'vmin': vmin, 'vmax': vmax})
            pltargs.update(**kwargs)
        im = ax[i].imshow(array[i], **pltargs)
        if title is not None:
            ax[i].text(0.05, 0.95, title[i], fontsize=fs,
                     #fontfamily='cm',
                     color='white',
                     verticalalignment='top', horizontalalignment='left', transform=ax[i].transAxes,
                     bbox=(dict(facecolor=bbox_info[3], alpha=alpha, edgecolor='none')))
        if pixel_measure is not None:
            distance_measure = int(4* pixel_measure/60)
            x0, y0 = 10*pixel_factor, 10 * pixel_factor
            x1, y1 = pixel_measure+10*pixel_factor, 10 *pixel_factor

            x2, y2 = 10*pixel_factor, 8*pixel_factor
            x3, y3 = 10*pixel_factor, 12*pixel_factor

            x4, y4 = pixel_measure+10*pixel_factor, 8*pixel_factor
            x5, y5 = pixel_measure+10*pixel_factor, 12*pixel_factor
            rect = patches.Rectangle(bbox_info[0], pixel_measure+bbox_info[1],
                                 bbox_info[2], facecolor=bbox_info[3], alpha=alpha)
            ax[i].add_patch(rect)
            ax[i].text(int(pixel_measure/2)+7*pixel_factor, 14*pixel_factor,
                       f"{distance_measure}'", fontsize=fs,
                       color='white')
            ax[i].plot([x0, x1], [y0, y1], color='white', linewidth=1)
            ax[i].plot([x2, x3], [y2, y3], color='white', linewidth=1)
            ax[i].plot([x4, x5], [y4, y5], color='white', linewidth=1)

    fig.subplots_adjust(right=0.84, wspace=0.04)
    cbar_ax = fig.add_axes([0.88, 0.38, 0.03, 0.225])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fs)

    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_file}.")
    else:
        plt.show()
    plt.close()

def plot_rgb(x, sat_min=[0, 0, 0], sat_max=[1, 1, 1], norm=None,
             sigma=None, log=False,
             title=None, pixel_measure=None, pixel_factor =1, fs=14, figsize=(8, 8), dpi=200,
             output_file=None, alpha=0.8,
             bbox_info= [(7, 4), 7,  30, 'black']):

    if sigma is not None:
        x = ju._smooth(sigma, x)
    if sat_min and sat_max is not None:
        x = ju._clip(x, sat_min, sat_max)
    if log:
        x = ju._non_zero_log(x)
    x = np.moveaxis(x, 0, -1)
    plot_data = ju._norm_rgb_plot(x)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    if norm is not None:
        ax.imshow(plot_data, origin="lower", norm=norm)
    else:
        ax.imshow(plot_data, origin="lower")

    if title is not None:
        ax.text(0.05, 0.95, title, fontsize=fs,
                 #fontfamily='cm',
                 color='white',
                 verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
                 bbox=(dict(facecolor=bbox_info[3], alpha=alpha, edgecolor='none')))
    if pixel_measure is not None:
        distance_measure = int(4 * pixel_measure / 60)
        x0, y0 = 10 * pixel_factor, 10 * pixel_factor
        x1, y1 = pixel_measure + 10 * pixel_factor, 10 * pixel_factor

        x2, y2 = 10 * pixel_factor, 8 * pixel_factor
        x3, y3 = 10 * pixel_factor, 12 * pixel_factor

        x4, y4 = pixel_measure + 10 * pixel_factor, 8 * pixel_factor
        x5, y5 = pixel_measure + 10 * pixel_factor, 12 * pixel_factor
        rect = patches.Rectangle(bbox_info[0], pixel_measure + bbox_info[1],
                                 bbox_info[2], facecolor=bbox_info[3], alpha=alpha)
        ax.add_patch(rect)
        ax.text(int(pixel_measure / 2) + 7 * pixel_factor, 14 * pixel_factor,
                   f"{distance_measure}'", fontsize=fs,
                   color='white')
        ax.plot([x0, x1], [y0, y1], color='white', linewidth=1)
        ax.plot([x2, x3], [y2, y3], color='white', linewidth=1)
        ax.plot([x4, x5], [y4, y5], color='white', linewidth=1)
    plt.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_file}.")
    else:
        plt.show()
    plt.close()


def plot_2d_gt_vs_rec_histogram(samples, operator_dict, diagnostics_path, response_dict,
                                reference_dict, base_filename=None, response=True, relative=False,
                                type='single', offset=0., plot_kwargs=None, fs=14, alpha=0.8, max_counts=None,
                                bbox_info= [(7, 4), 7,  30, 'black']):
    if 'pspec' in operator_dict.keys():
        operator_dict.pop('pspec')
    R = response_dict['R']
    if response is False:
        exp = response_dict['exposure']
        shape = exp(operator_dict[tuple(operator_dict)[0]](jft.mean(samples))).shape
        reshape = lambda x: np.tile(x, (shape[0], 1, 1, 1))
        R = lambda x, y: jft.Vector(
            {k: response_dict['mask_adj'](response_dict['mask'](reshape(x)))[0] for k in
            range(shape[0])})
    k = response_dict['kernel_arr']

    Rs_sample_dict = {key: [R(op(s), k) for s in samples] for key, op in operator_dict.items()}
    Rs_reference_dict = {key: R(ref, k) for key, ref in reference_dict.items()}

    for key in operator_dict.keys():
        res_list = []
        for Rs_sample in Rs_sample_dict[key]:
            for i, data_key in enumerate(Rs_sample.tree.keys()):
                if relative:
                    ref = Rs_reference_dict[key][data_key][Rs_reference_dict[key][data_key] != 0]
                    samp = Rs_sample[data_key][Rs_reference_dict[key][data_key] != 0]
                    res = np.abs(ref - samp) / ref
                else:
                    res = Rs_sample[data_key]
                    ref = Rs_reference_dict[key][data_key]
                if i == 0:
                    stacked_res = res.flatten()
                    stacked_ref = ref.flatten()
                else:
                    stacked_res = np.concatenate([stacked_res, res.flatten()]).flatten()
                    stacked_ref = np.concatenate([stacked_ref, ref.flatten()]).flatten()
            res_list.append(stacked_res)
        if type == 'single':
            res_1d_array_list = [np.stack(res_list).mean(axis=0).flatten()]
        elif type == 'sampled':
            res_1d_array_list = [sample for sample in res_list]
        else:
            raise NotImplementedError
        ref_list = len(res_1d_array_list) * [stacked_ref]
        if base_filename is not None:
            output_path = join(diagnostics_path, f'{base_filename}hist_{key}.png')
        else:
            output_path = None
        plot_sample_averaged_log_2d_histogram(x_array_list=ref_list,
                                              y_array_list=res_1d_array_list,
                                              output_path=output_path,
                                              offset=offset,
                                              fs=fs,
                                              alpha=alpha,
                                              max_counts= max_counts,
                                              bbox_info=bbox_info,
                                              **plot_kwargs)

def plot_sample_averaged_log_2d_histogram(x_array_list, x_label, y_array_list, y_label,
                                          x_lim=None, y_lim=None, bins=100, dpi=400,
                                          title=None, output_path=None, offset=None, fs=14,
                                          alpha=0.8, max_counts=None, bbox_info= [(7, 4), 7,  30, 'black']):

    if len(x_array_list) != len(y_array_list):
        raise ValueError('Need same number of samples for x- and y-axis.')

    # Add small offset to avoid logarithm of zero or negative values
    if offset is None:
        offset = 0.
    x_bins = np.logspace(np.log(np.min(np.nanmean(x_array_list, axis=0)) + offset),
                         np.log(np.max(np.nanmean(x_array_list, axis=0)) + offset), bins, base=np.e)
    y_bins = np.logspace(np.log(np.min(np.nanmean(y_array_list, axis=0)) + offset),
                         np.log(np.max(np.nanmean(y_array_list, axis=0)) + offset), bins, base=np.e)

    hist_list = []
    edges_x_list = []
    edges_y_list = []

    for i in range(len(x_array_list)):
        hist, edges_x, edges_y = np.histogram2d(x_array_list[i][~np.isnan(x_array_list[i])],
                                                y_array_list[i][~np.isnan(y_array_list[i])],
                                                bins=(x_bins, y_bins))
        hist_list.append(hist)
        edges_x_list.append(edges_x)
        edges_y_list.append(edges_y)

    # Create the 2D histogram
    fig, ax = plt.subplots(dpi=dpi)
    counts = np.mean(hist_list, axis=0)
    xedges = np.mean(edges_x_list, axis=0)
    yedges = np.mean(edges_y_list, axis=0)
    if max_counts is None:
        cax =plt.pcolormesh(xedges, yedges, counts.T, cmap=plt.cm.jet,
                   norm=LogNorm(vmin=1, vmax=np.max(counts)))
    else:
        cax =plt.pcolormesh(xedges, yedges, counts.T, cmap=plt.cm.jet,
                   norm=LogNorm(vmin=1, vmax=max_counts))
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.axhline(y=1, color='r', linestyle='-')
    plt.tight_layout()
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    if title is not None:
        ax.text(0.05, 0.95, title, fontsize=fs,
                 #fontfamily='cm',
                 color='white',
                 verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
                 bbox=(dict(facecolor=bbox_info[3], alpha=alpha, edgecolor='none')))
    if output_path is not None:
        plt.savefig(output_path)
        print(f"2D histogram saved as {output_path}.")
        plt.close()
    else:
        plt.show()

