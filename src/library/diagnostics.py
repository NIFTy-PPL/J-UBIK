import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

import nifty8 as ift

from .utils import save_rgb_image_to_fits, get_mask_operator
from .plot import plot_energy_slices, plot_result


def get_uncertainty_weighted_measure(sl,
                                     op=None,
                                     reference=None,
                                     output_dir_base=None,
                                     padder=None,
                                     mask_op=None,
                                     title='',
                                     abs=True,
                                     vmax=5):
    mpi_master = ift.utilities.get_MPI_params()[3]
    mean, var = sl.sample_stat(op)
    if mask_op is None:
        mask_op = ift.ScalingOperator(domain=mean.domain, factor=1)
    if reference is None:
        wgt_res = mean / var.sqrt()
    else:
        if abs:
            wgt_res = (mean - reference).abs() / var.sqrt()
        else:
            wgt_res = (reference-mean) / var.sqrt()

    wgt_res = mask_op.adjoint(wgt_res)
    if padder is None:
        padder = ift.ScalingOperator(domain=wgt_res.domain, factor=1)
    wgt_res = padder.adjoint(wgt_res)
    if mpi_master and output_dir_base is not None:
        with open(f'{output_dir_base}.pkl', 'wb') as file:
            pickle.dump(wgt_res, file)
        save_rgb_image_to_fits(wgt_res, output_dir_base,
                               overwrite=True, MPI_master=mpi_master)
        plt_args = {'cmap': 'seismic', 'vmin': -vmax, 'vmax': vmax}
        plot_result(wgt_res, f'{output_dir_base}.png', **plt_args)
    return wgt_res


def compute_noise_weighted_residuals(sample_list, op, reference_data, mask_op=None, output_dir=None,
                                     base_filename='nwr', abs=True, min_counts=0, plot_kwargs=None,
                                     n_bins=None):
    """ Computes the Poissonian noise-weighted residuals.
    The form of the residual is :math:`r = \\frac{s-d}{\\sqrt{s}}`, where s is the signal response
    and d the data.
    Args:
        sample_list: `nifty8.ResidualSampleList` posterior samples.
        op: `nifty8.Operator`, usually the signal response with respect to which the residuals
        are calculated.
        reference_data: `nifty8.Field`, the reference (masked or unmasked data) data.
        mask_op: `nifty8.Operator`, if the data is masked, the correspondent mask.
        output_dir: `str`, path to the output directory.
        base_filename: `str` base filename for the output plots. No extension.
        abs: `bool`, if True the absolute of the residuals is calculated.
        min_counts: `int` minimum number of data counts for which the residuals will be calculated.
        plot_kwargs: Keyword arguments for plotting.
        n_bins: `int` number of bins for histograms

    Returns:
        Posterior mean noise-weighted residuals
        If n_bins is not None:
            Noise-weigthed residuals distribution as a histogram
            Noise-weighted residuals plots in .fits and .png format

    """
    _, _, _, mpi_master = ift.utilities.get_MPI_params()
    sc = ift.StatCalculator()
    if n_bins is not None:
        hist_list = []

    for sample in sample_list.iterator():
        Rs = op.force(sample)
        nwr = (Rs - reference_data) / Rs.sqrt()
        if min_counts > 0:
            low_count_loc = reference_data.val < min_counts
            filtered_nwr = nwr.val.copy()
            filtered_nwr[low_count_loc] = np.nan
            nwr = ift.makeField(nwr.domain, filtered_nwr)
        if abs:
            nwr = nwr.abs()
        if mask_op is not None:
            nwr = mask_op.adjoint(nwr)
            flags = mask_op._flags
        sc.add(nwr)

        if n_bins is not None:
            unmasked_nwr = nwr.val[flags] if mask_op is not None else nwr.val
            hist, edges = np.histogram(unmasked_nwr.reshape(-1), bins=n_bins, range=(-5., 5.))
            hist_list.append(hist)

    mean_hist = np.mean(np.array(hist_list), axis=0)
    nwr_mean = sc.mean

    if mpi_master is not None and output_dir is not None:
        filename = output_dir + base_filename
        save_rgb_image_to_fits(nwr_mean, file_name=filename, overwrite=True, MPI_master=mpi_master)
        plot_kwargs.update({'cmap': 'seismic', 'vmin': -5., 'vmax': 5.})
        plot_result(nwr_mean, outname=f'{filename}.png', **plot_kwargs)
    if n_bins is None:
        return nwr_mean
    else:
        return nwr_mean, mean_hist, edges


def get_noise_weighted_residuals_from_file(sample_list_path,
                                           data_path,
                                           sky_op,
                                           response_op,
                                           mask_op,
                                           output_dir=None,
                                           abs=True,
                                           base_filename=None,
                                           min_counts=0,
                                           plot_kwargs=None,
                                           nbins=None):
    sl = ift.ResidualSampleList.load(sample_list_path)
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    return compute_noise_weighted_residuals(sl, response_op @ sky_op,
                                            mask_op(d),
                                            mask_op=mask_op,
                                            output_dir=output_dir,
                                            base_filename=base_filename,
                                            abs=abs,
                                            min_counts=min_counts,
                                            plot_kwargs=plot_kwargs,
                                            n_bins=nbins)


def signal_space_uwr_from_file(sl_path_base,
                               ground_truth_path,
                               sky_op,
                               padder,
                               mask_op,
                               output_dir_base=None,
                               title='signal space UWR'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(ground_truth_path, "rb") as f:
        gt = pickle.load(f)
    wgt_res = get_uncertainty_weighted_measure(sl, mask_op @ padder.adjoint @ sky_op.log(),
                                               mask_op(padder.adjoint(gt.log())),
                                               output_dir_base,
                                               title=title, abs=False, mask_op=mask_op)
    return wgt_res


def data_space_uwr_from_file(sl_path_base,
                             data_path,
                             sky_op,
                             response_op,
                             mask_op,
                             output_dir_base=None,
                             title='data space UWR'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(data_path, "rb") as f:
        d = pickle.load(f)
    wgt_res = get_uncertainty_weighted_measure(sl, response_op @ sky_op, mask_op(d),
                                               output_dir_base,
                                               mask_op=mask_op, title=title)
    return wgt_res


def signal_space_uwm_from_file(sl_path_base,
                               sky_op,
                               padder,
                               output_dir_base=None,
                               title='signal space UWM'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    wgt_mean = get_uncertainty_weighted_measure(sl, sky_op, output_dir_base=output_dir_base, padder=padder,
                                                title=title)
    return wgt_mean


def weighted_residual_distribution(sl_path_base,
                                   data_path,
                                   sky_op,
                                   response_op,
                                   mask_op,
                                   bins=200,
                                   output_dir_base=None,
                                   title='Weighted data residuals'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(data_path, "rb") as f:
        d = pickle.load(f)
    wgt_res = get_uncertainty_weighted_measure(sl, response_op @ sky_op, mask_op(d),
                                               output_dir_base=None, mask_op=mask_op, title=title)
    res = wgt_res.val.reshape(-1)
    _, edges = np.histogram(np.log10(res+1e-30), bins=bins)

    # _, edges = np.histogram(res, bins=np.logspace(np.log10(1e-10), np.log10(res.val.max), bins))

    # pl.hist(data, bins=np.logspace(np.log10(1e-10), np.log10(1.0), 50))
    # pl.gca().set_xscale("log")

    plt.hist(np.log10(res+1e-30), edges[0:])
    # plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.savefig(fname=output_dir_base + '.png')
    plt.close()
    return wgt_res


def plot_sky_flux_diagnostics(sl_path_base, gt_path, op, op_name, output_path, response_dict, bins,
                            x_lim, y_lim, levels):
    """ Plots distribution of reconstructed flux vs actual flux.
    ! This is thus only applicable for mock reconstructions
    Args:
        sl_path_base: path base to `nifty8.ResidualSampleList` posterior samples.
        gt_path: path to ground_truth of component
        op: `nifty8.Operator`, component of sky operator
        op_name: name of the operator compared
        output_path: saving path
        response_dict: xu.response_dict of mock reconstruction (mask, exposure_op, R)
        bins: number of bins for 2Dhist
        x_lim: limits of x_axis of plot
        y_lim: limits of y_axis of plot
        levels: levels of contours

    Returns:
        Noise-weighted residuals plots in .fits and .png format
    """
    with open(f"{sl_path_base}.p", "rb") as file:
        samples = pickle.load(file)

    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    full_mask = None
    mask = get_mask_operator(full_exposure)
    gt_1d_array = mask(gt).val.flatten()
    mean, var = None
    #mean, var = sl.sample_stat(op)
    rec_1d_array = mask(mean).val.flatten()

    x_bins = np.logspace(np.log(np.min(gt_1d_array)), np.log(np.max(gt_1d_array)), bins)
    y_bins = np.logspace(np.log(np.min(rec_1d_array)), np.log(np.max(rec_1d_array)), bins)

    # Create the 2D histogram
    fig, ax = plt.subplots(dpi=400)
    hist = ax.hist2d(gt_1d_array, rec_1d_array, bins=(x_bins, y_bins),
                     cmap=plt.cm.jet, norm=LogNorm())

    # Generate contour lines
    # xedges = hist[1]
    # yedges = hist[2]
    # X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    # Z = hist[0].T
    # contour = ax.contour(X, Y, Z, levels=levels, colors='white')

    # Add a colorbar
    cbar = fig.colorbar(hist[3], ax=ax)

    # Add line for comparison
    ax.plot(x_bins, x_bins, color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$I_{gt}$')
    ax.set_ylabel('$I_{rec}$')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title(f'Fluxes ({op_name}): Ground Truth vs. Reconstruction')
    plt.savefig(output_path)
    plt.close()


def plot_lambda_diagnostics(sl_path_base, gt_path, op, op_name, output_path, response_op, bins,
                            x_lim, y_lim, levels):
    """ Plots distribution of reconstructed lambda va. actual lambda
    ! This is thus only applicable for mock reconstructions
    Args:
        sl_path_base: path base to `nifty8.ResidualSampleList` posterior samples.
        gt_path: path to ground_truth of component
        op: `nifty8.Operator`, component of sky operator
        op_name: name of the operator compared
        output_path: saving path
        response_dict: xu.response_dict of mock reconstruction (mask, exposure_op, R)
        bins: number of bins for 2Dhist
        x_lim: limits of x_axis of plot
        y_lim: limits of y_axis of plot
        levels: levels of contours

    Returns:
        Noise-weighted residuals plots in .fits and .png format
    """
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    if gt.domain != op.target:
        raise ValueError(f'Ground truth domain and operator target do not fit together:'
                         f'Ground truth: {gt.domain}. op: {op.target}')

    gt_1d_array = response_op(gt).val.flatten()
    mean, var = sl.sample_stat(op)
    rec_1d_array = response_op(mean).val.flatten()

    x_bins = np.logspace(np.log(np.min(gt_1d_array)), np.log(np.max(gt_1d_array)), bins)
    y_bins = np.logspace(np.log(np.min(rec_1d_array)), np.log(np.max(rec_1d_array)), bins)

    # Create the 2D histogram
    fig, ax = plt.subplots(dpi=400)
    hist = ax.hist2d(gt_1d_array, rec_1d_array, bins=(x_bins, y_bins),
                     cmap=plt.cm.jet, norm=LogNorm())

    # Generate contour lines
    # xedges = hist[1]
    # yedges = hist[2]
    # X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    # Z = hist[0].T
    # contour = ax.contour(X, Y, Z, levels=levels, colors='white')

    # Add a colorbar
    cbar = fig.colorbar(hist[3], ax=ax)

    # Add line for comparison
    ax.plot(x_bins, x_bins, color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$I_{gt}$')
    ax.set_ylabel('$I_{rec}$')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title(f'Lambda ({op_name}): Ground Truth vs. Reconstruction')
    plt.savefig(output_path)
    plt.close()


def signal_space_weighted_residual_distribution(sl_path_base,
                                   ground_truth_path,
                                   sky_op,
                                   padder,
                                   mask_op,
                                   bins=200,
                                   output_dir_base=None,
                                   sample_diag=False,
                                   title='Weighted signal space residuals'):
    with open(ground_truth_path, "rb") as f:
        gt = pickle.load(f)
    sl = ift.ResidualSampleList.load(sl_path_base)
    wgt_res = get_uncertainty_weighted_measure(sl, mask_op @ padder.adjoint @ sky_op.log(),
                                               mask_op(padder.adjoint(gt.log())),
                                               output_dir_base, sample=sample_diag,
                                               title=title, abs=False, mask_op=mask_op)
    res = wgt_res.val.reshape(-1)
    range = (-5, 5)
    _, edges = np.histogram(np.log10(res+1e-30), bins=bins, range=range)
    plt.hist(np.log10(res+1e-30), edges[0:])
    # plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.savefig(fname=output_dir_base + '.png')
    plt.close()
    return wgt_res




