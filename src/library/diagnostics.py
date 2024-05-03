import jax
import numpy as np
from os.path import join
from jax.tree_util import tree_map
from jax import numpy as jnp

import nifty8.re as jft

from .utils import create_output_directory
from .plot import plot_result, plot_sample_averaged_log_2d_histogram, plot_histograms


def compute_uncertainty_weighted_residuals(samples,
                                           operator_dict,
                                           diagnostics_path,
                                           response_dict,
                                           reference_dict=None,
                                           base_filename=None,
                                           mask=False,
                                           abs=False,
                                           n_bins=None,
                                           range=None,
                                           log=True,
                                           plot_kwargs=None):
    """
    Computes uncertainty-weighted residuals (UWRs) given the position space sample list and
    plots them and the according histogram. Definition:
    :math:`uwrs = \\frac{s-gt}{\\sigma_{s}}`,
    where `s` is the signal, `gt` is the ground truth,
    and `sigma_{s}` is the standard deviation of `s`.

    Parameters
    ----------
    samples: nifty8.re.evi.Samples
        Position-space samples.
    operator_dict: dict
        Dictionary of operators for which the UWRs should be calculated.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    response_dict: dict
        Dictionary with response callables.
    reference_dict: dict, None
        Dictionary of reference arrays (e.g. ground truth for mock) to calculate the UWR.
        If the reference_dict is set to `None`, it is assumed to be zero everywhere and
        the uncertainty-weighted mean (UWM) are calculated.
    base_filename: str, None
        Base string of file name saved.
    mask: bool, False
        If true the output is masked by a mask generated from the intersection of all the exposures.
    abs: bool, False
        If true the absolute value of the residual is calculated and plotted.
    n_bins: int, None
        Number of bins in the histogram plotted. If None, no histogram is plotted.
    range: tuple, None
        Range of the histogram. If None, the range defaults to (-5, 5)
    log: bool, True
        If True, the log of the samples and reference are considered.
    plot_kwargs: dict, None
        Dictionary of plotting arguments for plot_results.

    Returns
    ----------
    res_dict: dict
        Dictionary of UWRs for each operator specified in the input operator_dict.
    """

    res_dict = {}
    for key, op in operator_dict.items():
        if key == "pspec":
            continue
        res_dict[key] = {}
        if reference_dict is None:
            reference_dict = {key: None}
        if key not in reference_dict:
            continue
        uwrs, exp_mask = _calculate_uwr(samples.samples, op, reference_dict[key], response_dict,
                                        abs=abs, exposure_mask=mask, log=log)
        uwrs = np.array(uwrs)
        masked_uwrs = uwrs.copy()
        masked_uwrs[~exp_mask] = np.nan
        res_dict[key]["uwrs"] = uwrs
        res_dict[key]["masked_uwrs"] = masked_uwrs
        if plot_kwargs is None:
            plot_kwargs = {}
        if 'cmap' not in plot_kwargs:
            plot_kwargs.update({'cmap': 'RdYlBu_r'})
        if 'vmin' not in plot_kwargs:
            plot_kwargs.update({'vmin': -5})
        if 'vmax' not in plot_kwargs:
            plot_kwargs.update({'vmax': 5})
        plot_result(masked_uwrs, output_file=join(diagnostics_path, f'{base_filename}{key}.png'),
                    **plot_kwargs)
        if n_bins:
            if range is None:
                range = (-5, 5)
            hist, edges = np.histogram(masked_uwrs.reshape(-1), bins=n_bins, range=range)
            title = plot_kwargs['title'] if plot_kwargs is not None else None
            plot_histograms(hist, edges, join(diagnostics_path, f'{base_filename}{key}_hist.png'),
                            title=title)
    return res_dict


def compute_noise_weighted_residuals(samples, operator_dict, diagnostics_path, response_dict,
                                     reference_data, base_filename=None, min_counts=0,
                                     response=True, mask_exposure=True, abs=False, n_bins=None,
                                     extent=None, plot_kwargs=None):
    """
    Computes noise-weighted residuals (NWRs) given the position space sample list and
    plots the according histogram. Definition:
    :math:`nwr = \\frac{Rs-d}{\\sqrt_{Rs}}`,
    where `s` is the signal, 'R' is the response and  `d` is the reference data.

    Parameters
    ----------
    samples: nifty8.re.evi.Samples
        Position-space samples.
    operator_dict: dict
        Dictionary of operators for which the NWRs should be calculated.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    response_dict: dict
        Dictionary with response callables.
    reference_data: dict, None
        Dictionary of reference arrays (e.g. data) to calculate the NWR.
    base_filename: str, None
        Base string of file name saved.
    abs: bool, False
        If true the absolute value of the residual is calculated and plotted.
    n_bins: int, None
        Number of bins in the histogram plotted. If None, no histogram is plotted.
    min_counts: `int`, minimum number of data counts for which the residuals will be calculated.
    response: Bool, True
        If True the response will be applied to the residuals.
    mask_exposure: bool, True
        If True the exposure mask will be applied to the residuals.
    abs: bool, False
        If true the absolute value of the residual is calculated and plotted.
    extent: tuple, None
        Range of the histogram. If None, the range defaults to (-5, 5)
    plot_kwargs: dict, None
        Dictionary of plotting keyword arguments for plot_result.

    Returns
    -------
    res_dict: dict
        Dictionary of noise-weighted residuals.
    """

    if base_filename is None:
        base_filename = ""
    res_dict = {}
    for key, op in operator_dict.items():
        if key == 'pspec':
            continue
        res_dict[key] = {}
        nwrs, mask = _calculate_nwr(samples.samples, op, reference_data, response_dict,
                                    abs=abs, min_counts=min_counts, exposure_mask=mask_exposure,
                                    response=response)
        masked_nwrs = nwrs.copy()
        masked_nwrs[mask] = np.nan
        res_dict[key]['nwrs'] = nwrs
        res_dict[key]['masked_nwrs'] = masked_nwrs
        if plot_kwargs is None:
            plot_kwargs = {}
        if 'cmap' not in plot_kwargs:
            plot_kwargs.update({'cmap': 'RdYlBu_r'})
        if 'vmin' not in plot_kwargs:
            plot_kwargs.update({'vmin': -5})
        if 'vmax' not in plot_kwargs:
            plot_kwargs.update({'vmax': 5})

        for id, i in enumerate(masked_nwrs):
            results_path = create_output_directory(join(diagnostics_path, f"tm_{id + 1}/{key}/"))
            if 'title' not in plot_kwargs:
                plot_kwargs.update({'title': f"NWR {key} - TM number {id + 1}"})
            plot_result(i,
                        output_file=join(results_path,
                                         f'{base_filename}{key}_tm{id + 1}_samples.png'),
                        adjust_figsize=True,
                        **plot_kwargs)
            plot_result(np.mean(i, axis=0),
                        output_file=join(results_path,
                                         f'{base_filename}{key}_tm{id + 1}_mean.png'),
                        **plot_kwargs)

            if n_bins is not None:
                if extent is None:
                    extent = (-5, 5)
                hist_func = lambda x: jnp.histogram(x.reshape(-1), bins=n_bins, range=extent)[0]
                edges_func = lambda x: jnp.histogram(x.reshape(-1), bins=n_bins, range=extent)[1]
                hist = tree_map(jax.vmap(hist_func, in_axes=0, out_axes=0), i)
                edges = tree_map(edges_func, nwrs)
                mean_hist = tree_map(lambda x: np.mean(x, axis=0), hist)
                plot_histograms(mean_hist, edges,
                                join(results_path, f'{base_filename}{key}_tm{id + 1}_hist.png'),
                                logy=False, title=f'NWR mean {key} - TM number {id + 1}')

    return res_dict


def _calculate_uwr(pos, op, ground_truth, response_dict,
                   abs=False, exposure_mask=True, log=True):
    op = jax.vmap(op)
    if ground_truth is None:
        print("No reference ground truth provided. "
              "Uncertainty-weighted residuals calculation defaults to uncertainty-weighted mean.")
    if log:
        ground_truth = jnp.log(ground_truth) if ground_truth is not None else 0.
        res = (jnp.mean(jnp.log(op(pos)), axis=0) - ground_truth) / jnp.std(jnp.log(op(pos)),
                                                                            axis=0, ddof=1)
    else:
        ground_truth = 0. if ground_truth is None else ground_truth
        res = (jnp.mean(op(pos), axis=0) - ground_truth) / jnp.std(op(pos), axis=0, ddof=1)
    if abs:
        res = jnp.abs(res)
    if exposure_mask:
        exposure = response_dict['exposure']
        exposure_mask = np.array(exposure(np.ones_like(res))).sum(axis=0) > 0
    return res, exposure_mask


def _calculate_nwr(pos, op, data, response_dict,
                   abs=False, min_counts=None, exposure_mask=True, response=True):
    if response:
        R = response_dict['R']
    else:
        R = lambda x: x

    adj_mask = response_dict['mask_adj']
    sqrt = lambda x: tree_map(jnp.sqrt, x)
    res = lambda x: (R(op(x)) - data) / sqrt(R(op(x)))
    res = jax.vmap(res, out_axes=1)
    res = np.array(jax.vmap(adj_mask, in_axes=1, out_axes=1)(res(pos))[0])
    if abs:
        res = np.abs(res)

    min_count_mask = None
    if min_counts is not None:
        masked_indices = lambda x: np.array(x < min_counts, dtype=float)
        masked_indices = tree_map(masked_indices, data)
        min_count_mask = lambda x: adj_mask(masked_indices)[0]
    if exposure_mask:
        exp_mask = lambda x: response_dict['exposure'](np.ones(op(x).shape)) == 0.
        if min_count_mask is not None:
            tot_mask = lambda x: np.logical_or(min_count_mask(x), exp_mask(x), dtype=bool)
        else:
            tot_mask = exp_mask
    else:
        tot_mask = min_count_mask if min_count_mask is not None else None
    if tot_mask is not None:
        tot_mask = jax.vmap(tot_mask, out_axes=1)
    return res, tot_mask(pos)


def plot_2d_gt_vs_rec_histogram(samples, operator_dict, diagnostics_path, response_dict,
                                reference_dict, base_filename=None, response=True, relative=False,
                                type='single', offset=0., plot_kwargs=None):
    """
    Plots the 2D histogram of reconstruction vs. ground-truth in either
    the data_space (if response_func = response) or the signal space (if response=False).
    If relative = True the relative error of the reconstruction is plotted vs. the ground-truth.
    It is possible to either plot the 2D histograms for reconstruction mean or instead the
    sample-averaged histograms (type=sampled).


    Parameters
    ----------
    samples: nifty8.re.evi.Samples
        nifty8.re.evi.Samples object containing the posterior samples of the reconstruction.
    operator_dict: dict
        Dictionary of operators for which the histogram should be plotted.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    response_dict: dict
        Dictionary containing the instrument response functions.
    reference_dict: dict, None
        Dictionary of reference arrays (e.g. ground-truth) to calculate the NWR.
    base_filename: str, None
        Base string of file name saved. If None, the plot is displayed and not saved.
    response: bool, True
        If True, the histogram for reconstruction vs. ground-truth is plotted.
    type: str, 'single'
        Either 'single' (default) taking the 2d histogram of the mean or 'sampled' to get
        the sample averaged histogram.
    offset: float, 0.
        Offset for the histogram (to prevent nan in log).
    relative: bool, False
        If False, the histogram for reconstruction vs. ground-truth is plotted.
        If True, the relative error vs. ground-truth histogram is generated.
    plot_kwargs: dict, None
        Dictionary of plotting arguments
    """
    # FIXME: CLEAN UP
    if 'pspec' in operator_dict.keys():
        operator_dict.pop('pspec')
    R = response_dict['R']
    if response is False:
        exp = response_dict['exposure']
        shape = exp(operator_dict[tuple(operator_dict)[0]](jft.mean(samples))).shape
        reshape = lambda x: np.tile(x, (shape[0], 1, 1))
        R = lambda x: jft.Vector(
            {k: response_dict['mask_adj'](response_dict['mask'](reshape(x)))[0] for k in
            range(shape[0])})

    Rs_sample_dict = {key: [R(op(s)) for s in samples] for key, op in operator_dict.items()}
    Rs_reference_dict = {key: R(ref) for key, ref in reference_dict.items()}

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
        if relative:
            for i, sample in enumerate(res_1d_array_list):
                res_1d_array_list[i] = np.abs(ref_list[i] - sample) / ref_list[i]
        if base_filename is not None:
            output_path = join(diagnostics_path, f'{base_filename}hist_{key}.png')
        else:
            output_path = None
        plot_sample_averaged_log_2d_histogram(x_array_list=ref_list,
                                              y_array_list=res_1d_array_list,
                                              output_path=output_path,
                                              offset=offset,
                                              **plot_kwargs)
