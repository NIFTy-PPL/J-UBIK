import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

import nifty8 as ift

from .utils import save_rgb_image_to_fits, get_mask_operator
from .plot import plot_energy_slices


def get_uncertainty_weighted_measure(sl,
                                     op=None,
                                     reference=None,
                                     output_dir_base=None,
                                     padder=None,
                                     mask_op=None,
                                     title=''):
    mpi_master = ift.utilities.get_MPI_params()[3]
    mean, var = sl.sample_stat(op)
    if mask_op is None:
        mask_op = ift.ScalingOperator(domain=mean.domain, factor=1)
    if reference is None:
        wgt_res = mean / var.sqrt()
    else:
        wgt_res = (mean - reference).abs() / var.sqrt()
    wgt_res = mask_op.adjoint(wgt_res)
    if padder is None:
        padder = ift.ScalingOperator(domain=wgt_res.domain, factor=1)
    wgt_res = padder.adjoint(wgt_res)
    if mpi_master and output_dir_base is not None:
        with open(f'{output_dir_base}.pkl', 'wb') as file:
            pickle.dump(wgt_res, file)
        save_rgb_image_to_fits(wgt_res, output_dir_base,
                               overwrite=True, MPI_master=mpi_master)
        plot_energy_slices(wgt_res, file_name=f'{output_dir_base}.png',
                           title=title, plot_kwargs={'norm': LogNorm()})
    return wgt_res


def compute_noise_weighted_residuals(sample_list, op, reference_data, mask_op=None, output_dir=None,
                                     base_filename='nwr', abs=True, plot_kwargs=None):
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
        plot_kwargs: Keyword arguments for plotting.

    Returns:
        Noise-weighted residuals plots in .fits and .png format

    """
    _, _, _, mpi_master = ift.utilities.get_MPI_params()
    sc = ift.StatCalculator()
    for sample in sample_list.iterator():
        Rs = op.force(sample)
        nwr = (Rs - reference_data) / Rs.sqrt()
        if abs:
            nwr = nwr.abs()
        if mask_op is not None:
            nwr = mask_op.adjoint(nwr)
        sc.add(nwr)

    nwr_mean = sc.mean

    if mpi_master is not None and output_dir is not None:
        filename = output_dir + base_filename
        plot_kwargs = {'title': 'Noise-weighted residuals'} if plot_kwargs is None else plot_kwargs
        save_rgb_image_to_fits(nwr_mean, file_name=filename, overwrite=True, MPI_master=mpi_master)
        plot_energy_slices(nwr_mean, file_name=f'{filename}.png', title='Noise-weighted residuals',
                           plot_kwargs={})
    return nwr_mean


def get_noise_weighted_residuals_from_file(sample_list_path,
                                           data_path,
                                           sky_op,
                                           response_op,
                                           mask_op,
                                           output_dir=None,
                                           abs=True,
                                           base_filename=None,
                                           plot_kwargs=None):
    sl = ift.ResidualSampleList.load(sample_list_path)
    with open(data_path, "rb") as f:
        d = pickle.load(f)
    wgt_res = compute_noise_weighted_residuals(sl, response_op @ sky_op, mask_op(d),
                                               mask_op=mask_op, output_dir=output_dir,
                                               base_filename=base_filename, abs=abs,
                                               plot_kwargs=plot_kwargs)
    return wgt_res


def signal_space_uwr_from_file(sl_path_base,
                               ground_truth_path,
                               sky_op,
                               padder,
                               output_dir_base=None,
                               title='signal space UWR'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(ground_truth_path, "rb") as f:
        gt = pickle.load(f)
    wgt_res = get_uncertainty_weighted_measure(sl, sky_op, gt, output_dir_base, padder=padder,
                                               title=title)
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


def plot_points_diagnostics(sl_path_base, gt_path, op, op_name, output_path, response_dict, bins,
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
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    if gt.domain != op.target:
        raise ValueError(f'Ground truth domain and operator target do not fit together:'
                         f'Ground truth: {gt.domain}. op: {op.target}')
    full_exposure = None
    for key, dict in response_dict.items():
        if full_exposure is None:
            full_exposure = dict['exposure_op'](ift.full(dict['exposure_op'].target, 1.))
        else:
            full_exposure = full_exposure + dict['exposure_op'](ift.full(dict['exposure_op'].target, 1.))
    mask = get_mask_operator(full_exposure)
    gt_1d_array = mask(gt).val.flatten()
    mean, var = sl.sample_stat(op)
    rec_1d_array = mask(mean).val.flatten()

    x_bins = np.logspace(np.log(np.min(gt_1d_array)), np.log(np.max(gt_1d_array)), bins)
    y_bins = np.logspace(np.log(np.min(rec_1d_array)), np.log(np.max(rec_1d_array)), bins)

    # Create the 2D histogram
    fig, ax = plt.subplots()
    hist = ax.hist2d(gt_1d_array, rec_1d_array, bins=(x_bins, y_bins), cmap=plt.cm.jet, norm=LogNorm())

    # Generate contour lines
    xedges = hist[1]
    yedges = hist[2]
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    Z = hist[0].T
    contour = ax.contour(X, Y, Z, levels=levels, colors='white')

    # Add a colorbar
    cbar = fig.colorbar(hist[3], ax=ax)

    # Add line for comparison
    ax.plot(x_bins, x_bins, color='black')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$I_{gt}$')
    ax.set_ylabel('$I_{rec}$')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title(f'Fluxes ({op_name}): Ground Truth vs. Reconstruction')
    plt.savefig(os.path.join(output_path, f'{op_name}_flux_diagnostics.png'))
    plt.close()


def signal_space_weighted_residual_distribution(sl_path_base,
                                   ground_truth_path,
                                   sky_op,
                                   padder,
                                   bins=200,
                                   output_dir_base=None,
                                   title='Weighted signal space residuals'):
    with open(ground_truth_path, "rb") as f:
        gt = pickle.load(f)
    sl = ift.ResidualSampleList.load(sl_path_base)
    wgt_res = get_uncertainty_weighted_measure(sl, sky_op, reference=gt, padder=padder,
                                               output_dir_base=None, title=title)
    res = wgt_res.val.reshape(-1)
    _, edges = np.histogram(np.log10(res+1e-30), bins=bins)

    plt.hist(np.log10(res+1e-30), edges[0:])
    # plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.savefig(fname=output_dir_base + '.png')
    plt.close()
    return wgt_res




