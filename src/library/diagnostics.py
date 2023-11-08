import numpy as np
import pickle
from os.path import join
from functools import partial

import nifty8 as ift

from .utils import save_rgb_image_to_fits, get_config, create_output_directory
from .plot import plot_result, plot_sample_averaged_log_2d_histogram, plot_histograms
from .sky_models import create_sky_model_from_config
from .response import build_callable_from_exposure_file, build_exposure_function,\
    build_readout_function
from .data import load_masked_data_from_pickle, load_erosita_masked_data


def _build_full_mask_func_from_exp_files(exposure_file_names,
                                         threshold):
    def _set_zero(x, exp):
        x[exp == 0] = 0
        return x

    def _builder(exposures, threshold):
        if threshold < 0:
            raise ValueError("threshold should be positive!")
        if threshold is not None:
            exposures[exposures < threshold] = 0
        summed_exp = np.sum(exposures, axis=0)
        return partial(_set_zero, exp=summed_exp)

    return build_callable_from_exposure_file(_builder, exposure_file_names, threshold=threshold)


def get_diagnostics_from_file(diagnostic_builder,
                              diagnostics_path,
                              sl_path_base,
                              pos_path_base,
                              config_path,
                              output_operators_keys,
                              **kwargs):
    # Build diagnostics directory
    create_output_directory(diagnostics_path)

    # Get config info
    cfg = get_config(config_path)

    # Create sky operators
    sky_dict = create_sky_model_from_config(config_path)
    sky_dict.pop('pspec')

    # Load position space sample list
    with open(f"{sl_path_base}.p", "rb") as file:
        samples = pickle.load(file)
    with open(f"{pos_path_base}.p", "rb") as file:
        pos = pickle.load(file)

    lat_sp_sl = list(samples.at(pos))
    pos_sp_sample_dict = {key: np.stack([op(lat_sp_sample) for lat_sp_sample in lat_sp_sl])
                          for key, op in sky_dict.items() if key in output_operators_keys}

    diagnostic_builder(pos_sp_sample_dict, diagnostics_path, cfg, **kwargs)


def compute_uncertainty_weighted_residuals(pos_sp_sample_dict,
                                           diagnostics_path,
                                           cfg,
                                           reference_dict=None,
                                           output_dir_base=None,
                                           mask=False,
                                           abs=False,
                                           n_bins=None,
                                           range=None,
                                           log=True,
                                           plot_kwargs=None):
    """
    Computes and plots uncertainty-weighted residuals, defined as
    :math:`uwr = \\frac{s-gt}{\\sigma_{s}}`, where `s` is the signal, `gt` is the ground truth,
    and `sigma_{s}` is the standard deviation of `s`.
    Args:
        sample_list: `nifty8.ResidualSampleList`, posterior samples.
        op: `nifty8.Operator`, the operator to compute statistics on. Defaults to None.
        reference: `nifty8.Field`, the reference ground truth. Defaults to None.
        output_dir_base: `str`, the base directory for output files. Defaults to None.
        padder: `nifty8.Operator`, the padding operator. Defaults to None.
        mask_op: `nifty8.Operator`, the mask operator. Defaults to None.
        abs: `bool`, whether to compute the absolute difference between the mean and reference.
        Defaults to False.
        n_bins: `int`, the number of bins for the histogram. Defaults to None.
        range: `tuple`, the range of values for the histogram. Defaults to (-5, 5).
        plot_kwargs: `dict`, additional keyword arguments for plotting. Defaults to None.
    Returns:
        `nifty8.Field`, the uncertainty-weighted residuals
        Additionally, if n_bins is not None:
            `ndarray`, uncertainty-weighted-residuals distribution as a histogram.
            `ndarray`, edges of the uncertainty-weighted-residuals distribution as a histogram.
    """

    mpi_master = ift.utilities.get_MPI_params()[3]
    uwr_dict = {}
    if log:
        pos_sp_sample_dict = {key: np.log(value) for key, value in pos_sp_sample_dict.items()}
        if reference_dict is not None:
            reference_dict = {key: np.log(value) for key, value in reference_dict.items()}
    for key, pos_sp_sl in pos_sp_sample_dict.items():
        mean = pos_sp_sl.mean(axis=0)
        std = pos_sp_sl.std(axis=0, ddof=1)

        if reference_dict is None:
            print("No reference ground truth provided. "
                "Uncertainty weighted residuals calculation defaults to uncertainty-weighted mean.")
            return mean / std

        if abs:
            uwr = (mean - reference_dict[key]).abs() / std
        else:
            uwr = (reference_dict[key]-mean) / std

        if mask:
            tel_info = cfg['telescope']
            file_info = cfg['files']
            exposure_file_names = [join(file_info['obs_path'], f'{key}_' + file_info['exposure'])
                                  for key in tel_info['tm_ids']]
            full_mask_func = _build_full_mask_func_from_exp_files(exposure_file_names,
                                                                  tel_info['exp_cut'])
            uwr = full_mask_func(uwr)
        uwr_dict[key] = uwr

        if mpi_master and output_dir_base is not None:
            with open(f'{output_dir_base}.pkl', 'wb') as file:
                pickle.dump(uwr, file)
            # FIXME save_rgb-image_to_fits needs to be transferred to jubix here
            # save_rgb_image_to_fits(uwr, output_dir_base, overwrite=True, MPI_master=mpi_master)
            if plot_kwargs is None:
                plot_kwargs = {}
            if 'cmap' not in plot_kwargs:
                plot_kwargs.update({'cmap': 'seismic'})
            if 'vmin' not in plot_kwargs:
                plot_kwargs.update({'vmin': -5})
            if 'vmax' not in plot_kwargs:
                plot_kwargs.update({'vmax': 5})
            plot_result(uwr, output_file=join(diagnostics_path, f'{output_dir_base}{key}.png'),
                        **plot_kwargs)

        if n_bins is not None and output_dir_base is not None:
            if range is None:
                range = (-5, 5)
            hist, edges = np.histogram(uwr.reshape(-1), bins=n_bins, range=range)
            plot_histograms(hist, edges,
                            join(diagnostics_path, f'{output_dir_base}hist_{key}.png'),
                            title=plot_kwargs['title'])
    return uwr_dict


def compute_noise_weighted_residuals(sample_list, op, reference_data, mask_op=None, output_dir=None,
                                     base_filename='nwr', abs=False, min_counts=0, plot_kwargs=None,
                                     n_bins=None, range=None):
    """ Computes the Poissonian noise-weighted residuals.
    The form of the residual is :math:`nwr = \\frac{s-d}{\\sqrt{s}}`, where s is the signal response
    and d the data.
    Args:
        sample_list: `nifty8.ResidualSampleList`, posterior samples.
        op: `nifty8.Operator`, usually the signal response with respect to which the residuals
        are calculated.
        reference_data: `nifty8.Field`, the reference (masked or unmasked data) data.
        mask_op: `nifty8.Operator`, if the data is masked, the correspondent mask.
        output_dir: `str`, path to the output directory.
        base_filename: `str`, base filename for the output plots. No extension.
        abs: `bool`, if True the absolute of the residuals is calculated. Defaults to False.
        min_counts: `int`, minimum number of data counts for which the residuals will be calculated.
        plot_kwargs: `dict`, additional keyword arguments for plotting.
        n_bins: `int`, number of bins for histograms.
        range: `tuple`, (min, max) range of the histogram, defaults to (-5, 5).

    Returns:
        `nifty8.Field`, posterior mean noise-weighted residuals
        Additionally, if n_bins is not None:
            `ndarray`, posterior mean of noise-weigthed residuals distribution as a histogram.
            `ndarray`, edges of the posterior mean of noise-weigthed residuals distribution as a
            histogram.

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
            if range is None:
                range = (-5, 5)
            hist, edges = np.histogram(unmasked_nwr.reshape(-1), bins=n_bins, range=range)
            hist_list.append(hist)

    if n_bins is not None:
        mean_hist = np.mean(np.array(hist_list), axis=0)
    nwr_mean = sc.mean

    if mpi_master is not None and output_dir is not None:
        import os.path.join as join
        filename = join(output_dir, base_filename)
        save_rgb_image_to_fits(nwr_mean, file_name=filename, overwrite=True, MPI_master=mpi_master)
        if plot_kwargs is None:
            plot_kwargs = {}
        if 'cmap' not in plot_kwargs:
            plot_kwargs.update({'cmap': 'seismic'})
        if 'vmin' not in plot_kwargs:
            plot_kwargs.update({'vmin': -5})
        if 'vmax' not in plot_kwargs:
            plot_kwargs.update({'vmax': 5})
        plot_result(nwr_mean, outname=f'{filename}.png', **plot_kwargs)

    if n_bins is None:
        return nwr_mean
    else:
        return nwr_mean, mean_hist, edges


def get_noise_weighted_residuals_from_file(sample_list_path,
                                           config_path,
                                           output_operators_keys,
                                           output_dir=None,
                                           abs=False,
                                           base_filename=None,
                                           min_counts=0,
                                           plot_kwargs=None,
                                           nbins=None,
                                           range=(-5, 5)):
    """
    Retrieves and plots noise-weighted residuals from sample list file.

    Args:
        sample_list_path: `str`, the path to the sample list file.
        data_path: `str`, the path to the data file.
        sky_op: `nifty8.Operator`, the operator representing the sky.
        response_op: `nifty8.Operator`, the operator representing the response.
        mask_op: `nifty8.Operator`, the operator representing the mask.
        output_dir: `str`, Optional, the output directory path. Defaults to None.
        abs: `bool`, Optional, whether to compute absolute values. Defaults to False.
        base_filename: `str`, Optional, the base filename for output files. Defaults to None.
        min_counts: `int`, Optional, the minimum counts. Defaults to 0.
        plot_kwargs: `dict`, Optional, additional keyword arguments for plotting. Defaults to None.
        nbins: `int`, Optional, the number of bins for plotting. Defaults to None.
        If none, no histogram is returned
        range: `tuple`, Optional, (min, max) range of the histogram, defaults to (-5, 5).

    Returns:
        If nbins is None:
            `nifty8.Field`: The noise-weighted residuals.
        If nbins is not None:
            `ndarray`: The noise-weighted residuals.
            `ndarray`: The histogram of the noise-weighted residuals.
            `ndarray`: The edges of the histogram of the noise-weighted residuals
    """


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
                                            n_bins=nbins,
                                            range=range)


# FIXME Docstring
def plot_2d_gt_vs_rec_histogram(sl_path_base, pos_path_base, gt_path, op, op_name, response_func, pad, bins,
                                  output_path, x_lim=None, y_lim=None, x_label='', y_label='',
                                  dpi=400, title='', type='single', relative=False):
    """ Plots the 2D histogram of reconstruction vs. ground-truth in either
    the data_space (if response = full response) or the signal space (if response = mask).

    Parameters:
    -----------
    sl_path_base : `str`
        Path to ift.ResidualSampleList
    gt_path : `str`
        Path to pickled ground-truth field
    op : `ift.Operator`
        Sky operator for which the diagnostic is performed
    op_name : `str`
        name of the operator
    response : `ift.Operator`
        Operator which is applied to the ground-truth and reconstruction field
        before generating the histogram (if response = full response: data space,
        if response = mask: signal space)
    pad : `ift.Operator`
        padder of the sky model
    bins : `int`
        Number of bins of the 2D-histogram
    output_path : `str`, optional
        Output directory for the plot. If None (Default) the plot is not saved.
    xlim : `float`
        xlim of the x-axis of the plot (Default: None)
    ylim : `float`
        ylim of the y-axis of the plot (Default: None)
    x_label: `str`, optional
        x-axis label (Default: '')
    y_label: `str`, optional
        y-axis label (Default: '')
    dpi : `int`, optional
        Resolution of the figure
    title : `str`, optional
        Title of the 2D histogram
    type : `str`, optional
        Type of the histogram. Can either be 'single' for a 2D histogram for single fields
        or 'sampled' for a sampled histogram.
    relative: `bool`
        If True the relative distance is plotted on the y-axis of the histogramm.
        If False the x-axis shows the reconstruction is plotted on the y-axis.
        In either case the x-axis shows the ground truth.

    Returns:
    --------
    None
    """
    # Load position space sample list
    with open(f"{sl_path_base}.p", "rb") as file:
        samples = pickle.load(file)
    with open(f"{pos_path_base}.p", "rb") as file:
        pos = pickle.load(file)

    lat_sp_sl = list(samples.at(pos))
    pos_sp_sl = [op(lat_sp_sample) for lat_sp_sample in lat_sp_sl]

    # Load ground truth
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    gt_1d_array = response_func(gt)
    if relative:
        masked_gt_1d_array = gt_1d_array[gt_1d_array != 0]
        gt_1d_array_list = [masked_gt_1d_array]
    else:
        gt_1d_array_list = [gt_1d_array]
    if type == 'single':
        mean, _ = sl.sample_stat(op)
        mean = pad.adjoint(mean)
        res_1d_array = response(mean).val.flatten()
        if relative:
            res_1d_array = res_1d_array[gt_1d_array != 0]
            res_1d_array = np.abs((gt_1d_array_list[0]-res_1d_array)/gt_1d_array_list[0])
        res_1d_array_list = [res_1d_array]
    elif type == 'sampled':
        res_1d_array_list = []
        for sample in sl.iterator():
            res = op.force(sample)
            res_1d_array = response(res).val.flatten
            if relative:
                res_1d_array = res_1d_array[gt_1d_array != 0]
                res_1d_array = np.abs((gt_1d_array_list[0] - res_1d_array) / gt_1d_array_list[0])
            res_1d_array_list.append(res_1d_array)
        gt_1d_array_list = len(res_1d_array_list)*gt_1d_array_list
    else:
        raise NotImplementedError
    plot_sample_averaged_log_2d_histogram(x_array_list=gt_1d_array_list, x_label=x_label,
                                            y_array_list=res_1d_array_list, y_label=y_label,
                                            x_lim=x_lim, y_lim=y_lim,
                                            bins=bins, dpi=dpi,
                                            title=f'{op_name}: {title}',
                                            output_path=output_path)

