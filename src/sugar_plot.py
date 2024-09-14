# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from os.path import join

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np
from jax import random, linear_transpose
from jax.tree_util import tree_map

from .diagnostics import calculate_uwr, calculate_nwr
from .instruments.erosita.erosita_response import (
    build_erosita_response_from_config)
from .plot import (plot_result, plot_sample_averaged_log_2d_histogram,
                   plot_histograms, plot_rgb)
from .sky_models import SkyModel
from .utils import get_stats, create_output_directory, get_config


def plot_pspec(pspec, shape, distances,
               sample_list, output_directory,
               iteration=None, dpi=300,
               directory_prefix="spatial_"
               ):
    """
    Plots the power spectrum from a list of samples.

    Parameters:
    -----------
    pspec : callable
        The power spectrum function to be applied
        to the samples.
    shape : tuple[int]
        The shape of the grid or field for which the
        power spectrum is computed.
    distances : Union[float, tuple[float]]
        The distances in the grid corresponding
        to each axis.
    sample_list : nifty8.re.evi.Samples
        A list of samples to be used for generating the
        power spectrum.
    output_directory : str
        The directory where the plot files will be saved.
    iteration : int, optional
        The global iteration number value.
        Defaults to None, which uses 0.
    dpi : int, optional
        The resolution of the plot in dots per inch.
        Defaults to 300.
    directory_prefix : str, optional
        A prefix for the directory name where
        plots are saved.
        Defaults to "spatial_".

    Returns:
    --------
    None
    """
    if iteration is None:
        iteration = 0
    results_path = create_output_directory(join(output_directory,
                                                f"{directory_prefix}pspec"))
    samples = jax.vmap(pspec)(sample_list.samples)
    filename_samples = join(results_path, f"samples_{iteration}.png")
    from nifty8.re.correlated_field import get_fourier_mode_distributor
    _, unique_modes, _ = get_fourier_mode_distributor(shape, distances)

    plt.plot(unique_modes, jft.mean(samples), label="mean")
    for s in samples:
        plt.plot(unique_modes, s, alpha=0.5, color='k')
    plt.loglog()
    plt.savefig(filename_samples, dpi=dpi)
    plt.cla()
    plt.clf()
    plt.close()
    print(f"Power spectrum saved as {filename_samples}.")


def plot_sample_and_stats(output_directory,
                          operators_dict,
                          sample_list,
                          iteration=None,
                          log_scale=True,
                          colorbar=True,
                          dpi=300,
                          plotting_kwargs=None,
                          rgb_min_sat=None, rgb_max_sat=None,
                          plot_samples=True):
    """
    Plots operator samples and statistics from a sample list.

    Parameters:
    -----------
    output_directory : str
        The directory where the plot files will be saved.
    operators_dict : dict[callable]
        A dictionary containing operators.
    sample_list : nifty8.re.evi.Samples
        A list of samples.
    iteration : int, optional
        The global iteration number value. Defaults to None.
    log_scale : bool, optional
        Whether to use a logarithmic scale. Defaults to True.
    colorbar : bool, optional
        Whether to show a colorbar. Defaults to True.
    dpi : int, optional
        The resolution of the plot. Defaults to 100.
    plotting_kwargs : dict, optional
        Additional plotting keyword arguments. Defaults to None.
    rgb_min_sat : float, optional
        Absolute minimal saturation for individual color channels.
    rgb_max_sat : float, optional
        Absolute maximal saturation for individual color channels.
        For example, 0.5 clips the plot at half the intensity.
    plot_samples : bool, optional
        Whether to plot the samples. Defaults to True.

    Returns:
    --------
    None
    """

    if len(sample_list) == 0:
        sample_list = [sample_list.pos]
    if iteration is None:
        iteration = 0
    if plotting_kwargs is None:
        plotting_kwargs = {}

    for key in operators_dict:
        op = operators_dict[key]
        n_samples = len(sample_list)
        operator_samples = np.array([op(s) for s in sample_list])
        e_length = operator_samples[0].shape[0]

        # Create output directories
        results_path = create_output_directory(join(output_directory, key))
        stats_result_path = create_output_directory(join(results_path, "stats"))
        samples_result_path = create_output_directory(join(results_path,
                                                           "samples",
                                                           f"iteration_"
                                                           f"{iteration}"))
        if e_length == 3:
            rgb_result_path_samples = create_output_directory(
                join(results_path, "rgb", "samples",
                     f"iteration_{iteration}"))
            rgb_result_path_stats = create_output_directory(
                join(results_path, "rgb", "stats"))

        filename_mean = join(stats_result_path, f"mean_it_{iteration}.png")
        filename_std = join(stats_result_path, f"std_it_{iteration}.png")

        # Plot samples
        if plot_samples:
            for i in range(n_samples):
                filename_samples = join(samples_result_path,
                                        f"sample_{i + 1}_it_{iteration}.png")
                title = [f"Energy {ii + 1}" for ii in range(e_length)]
                plotting_kwargs.update({'title': title})
                plot_result(operator_samples[i],
                            output_file=filename_samples,
                            logscale=log_scale,
                            colorbar=colorbar,
                            dpi=dpi,
                            adjust_figsize=True,
                            **plotting_kwargs)

                # :TODO enable arbitrary multi-frequency rgb plotting
                if e_length == 3:
                    # Plot RGB
                    rgb_filename = join(rgb_result_path_samples,
                                        f"sample_{i + 1}_{iteration}_rgb")
                    plot_rgb(operator_samples[i], rgb_filename,
                             sat_min=rgb_min_sat,
                             sat_max=rgb_max_sat)
                    plot_rgb(operator_samples[i], rgb_filename + "_log",
                             sat_min=rgb_min_sat,
                             sat_max=None, log=True)

        # Plot statistics
        if 'n_rows' in plotting_kwargs:
            plotting_kwargs.pop('n_rows')
        if 'n_cols' in plotting_kwargs:
            plotting_kwargs.pop('n_cols')
        if 'figsize' in plotting_kwargs:
            plotting_kwargs.pop('figsize')
        if 'title' in plotting_kwargs:
            plotting_kwargs.pop('title')

        if len(sample_list) > 1:
            mean, std = get_stats(sample_list, op)
            title = [f"Posterior mean (energy {ii + 1})" for ii in
                     range(e_length)]
            plot_result(mean, output_file=filename_mean, logscale=log_scale,
                        colorbar=colorbar, title=title, dpi=dpi,
                        **plotting_kwargs)
            title = [f"Posterior std (energy {ii + 1})" for ii in
                     range(e_length)]
            plot_result(std, output_file=filename_std, logscale=log_scale,
                        colorbar=colorbar, title=title, dpi=dpi,
                        **plotting_kwargs)

            if e_length == 3:
                rgb_name = join(rgb_result_path_stats,
                                f"_mean_it_{iteration}_rgb")
                plot_rgb(mean, rgb_name, sat_min=rgb_min_sat,
                         sat_max=rgb_max_sat)
                plot_rgb(mean, rgb_name + "_log", sat_min=rgb_min_sat,
                         sat_max=None, log=True)


def plot_erosita_priors(key,
                        n_samples,
                        config_path,
                        priors_dir,
                        signal_response=False,
                        plotting_kwargs=None,
                        common_colorbar=False,
                        log_scale=True,
                        adjust_figsize=False):
    """
    Plots prior samples for the signal components of the sky
    through the eROSITA signal response from the config file.

    Parameters:
    ----------
        key : np.ndarray
            The random key for reproducibility.
        n_samples : int
            The number of samples to generate.
        config_path : str
            The path to the config file.
        priors_dir : str
            The directory to save the priors plots.
        signal_response : bool, optional
            Whether to pass the signal through the eROSITA response.
            If False, only the signal will be plotted,
            without passing it through the eROSITA response.
        plotting_kwargs : dict, optional
            Additional keyword arguments for plotting.
        common_colorbar : bool, optional
            Whether to use a common colorbar for all plots.
        log_scale : bool, optional
            Whether to use a logarithmic scale for the plots.
        adjust_figsize : bool, optional
            Whether to automatically adjust the figure
            size aspect ratio for the plots.

    Returns:
    -------
        None
    """
    priors_dir = create_output_directory(priors_dir)
    cfg = get_config(config_path)  # load config

    e_min = cfg['grid']['energy_bin']['e_min']
    e_max = cfg['grid']['energy_bin']['e_max']

    if plotting_kwargs is None:
        plotting_kwargs = {}

    sky_model = SkyModel(config_path)
    _ = sky_model.create_sky_model()
    plottable_ops = sky_model.sky_model_to_dict()
    positions = []
    for _ in range(n_samples):
        key, subkey = random.split(key)
        positions.append(jft.random_like(subkey, plottable_ops['sky'].domain))

    plottable_samples = plottable_ops.copy()
    sample_dirs = [join(priors_dir, f'sample_{i}/') for i in range(n_samples)]

    for i, pos in enumerate(positions):
        sample_dir = create_output_directory(sample_dirs[i])
        filename_base = sample_dir + 'priors_{}.png'
        for key, val in plottable_samples.items():
            plot_result(val(pos), output_file=filename_base.format(key),
                        logscale=log_scale, adjust_figsize=adjust_figsize,
                        title=[f'E_min={emin}, E_max={emax}' for emin, emax in
                               zip(e_min, e_max)],
                        common_colorbar=common_colorbar, **plotting_kwargs)

    # TODO: load from pickle, when response pickle is enabled
    if signal_response:
        tm_ids = cfg['telescope']['tm_ids']
        n_modules = len(tm_ids)

        spix = cfg['grid']['sdim']
        epix = cfg['grid']['edim']
        response_dict = build_erosita_response_from_config(config_path)

        mask_adj = linear_transpose(response_dict['mask'],
                                    np.zeros(
                                        (n_modules, epix, spix, spix)))
        # TODO: unify cases
        if 'kernel' in response_dict:
            R = lambda x: mask_adj(
                response_dict['R'](x, response_dict['kernel']))[0]
        else:
            R = lambda x: mask_adj(response_dict['R'](x))[0]

        for i, pos in enumerate(positions):
            for key, val in plottable_samples.items():
                tmp = R(val(pos))
                for id, samps in enumerate(tmp):
                    tm_id = tm_ids[id]
                    res_path = join(sample_dirs[i], f'tm{tm_id}/')
                    create_output_directory(res_path)
                    filename = join(res_path, f'sr_priors')
                    filename += '_{}.png'
                    plot_result(samps, output_file=filename.format(key),
                                logscale=log_scale,
                                title=[f'E_min={emin}, E_max={emax}' for
                                       emin, emax in
                                       zip(e_min, e_max)],
                                common_colorbar=common_colorbar,
                                adjust_figsize=adjust_figsize)


def plot_uncertainty_weighted_residuals(samples,
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
    Plots uncertainty-weighted residuals (UWRs) given the position space
    sample list and
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
        Dictionary of reference arrays (e.g. ground truth for mock) to
        calculate the UWR.
        If the reference_dict is set to `None`, it is assumed to be zero
        everywhere and the uncertainty-weighted mean (UWM) are calculated.
    base_filename: str, None
        Base string of file name saved.
    mask: bool, False
        If true the output is masked by a mask generated from the
        intersection of all the exposures.
    abs: bool, False
        If true the absolute value of the residual is calculated and plotted.
    n_bins: int, None
        Number of bins in the histogram plotted.
        If None, no histogram is plotted.
    range: tuple, None
        Range of the histogram. If None, the range defaults to (-5, 5)
    log: bool, True
        If True, the log of the samples and reference are considered.
    plot_kwargs: dict, None
        Dictionary of plotting arguments for plot_results.

    Returns
    ----------
    res_dict: dict
        Dictionary of UWRs for each operator specified in the input
        operator_dict.
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
        if reference_dict[key] is None:
            print(f"No reference ground truth provided for {key}. "
                  "Uncertainty-weighted residuals calculation defaults to "
                  "uncertainty-weighted mean.")

        uwrs, exp_mask = calculate_uwr(samples.samples, op, reference_dict[key],
                                       response_dict, abs=abs,
                                       exposure_mask=mask, log=log)
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
        plot_result(masked_uwrs, output_file=join(diagnostics_path,
                                                  f'{base_filename}{key}.png'),
                    **plot_kwargs)
        if n_bins:
            if range is None:
                range = (-5, 5)
            hist, edges = np.histogram(masked_uwrs.reshape(-1), bins=n_bins,
                                       range=range)
            title = plot_kwargs['title'] if plot_kwargs is not None else None
            plot_histograms(hist, edges, join(diagnostics_path,
                                              f'{base_filename}{key}_hist.png'),
                            title=title)
    return res_dict


def plot_noise_weighted_residuals(samples,
                                  operator_dict,
                                  diagnostics_path,
                                  response_dict,
                                  reference_data,
                                  base_filename=None,
                                  min_counts=0,
                                  response=True,
                                  mask_exposure=True,
                                  abs=False,
                                  n_bins=None,
                                  extent=(-5, 5),
                                  plot_kwargs=None):
    """
    Plots noise-weighted residuals (NWRs) given the position space sample
    list and
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
        Number of bins in the histogram plotted.
        If None, no histogram is plotted.
    min_counts: `int`,
        Minimum number of data counts for which the residuals
        will be calculated.
    response: Bool, True
        If True the response will be applied to the residuals.
    mask_exposure: bool, True
        If True the exposure mask will be applied to the residuals.
    abs: bool, False
        If true the absolute value of the residual is calculated and plotted.
    extent: tuple, None
        Range of the histogram.
        Default is (-5, 5).
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
        nwrs, mask = calculate_nwr(samples.samples, op, reference_data,
                                   response_dict,
                                   abs=abs, min_counts=min_counts,
                                   exposure_mask=mask_exposure,
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
            results_path = create_output_directory(
                join(diagnostics_path, f"tm_{id + 1}/{key}/"))
            if 'title' not in plot_kwargs:
                plot_kwargs.update({'title': f"NWR {key} - TM number {id + 1}"})
            for sample in range(i.shape[0]):
                plot_result(i[sample],
                            output_file=join(results_path,
                                             f'{base_filename}{key}_tm'
                                             f'{id + 1}_sample_{sample}.png'),
                            adjust_figsize=True,
                            **plot_kwargs)
            plot_result(np.mean(i, axis=0),
                        output_file=join(results_path,
                                         f'{base_filename}{key}_tm'
                                         f'{id + 1}_mean.png'),
                        **plot_kwargs)

            if n_bins is not None:
                hist_func = lambda x: jnp.histogram(x.reshape(-1),
                                                    bins=n_bins,
                                                    range=extent)[0]
                edges_func = lambda x: jnp.histogram(x.reshape(-1),
                                                     bins=n_bins,
                                                     range=extent)[1]
                hist = tree_map(jax.vmap(hist_func, in_axes=0, out_axes=0), i)
                edges = tree_map(edges_func, masked_nwrs)
                mean_hist = tree_map(lambda x: np.mean(x, axis=0), hist)
                hist_filename = join(results_path,
                                     f'{base_filename}{key}_tm'
                                     f'{id + 1}_hist.png')
                plot_histograms(mean_hist, edges,
                                hist_filename,
                                logy=False,
                                title=f'NWR mean {key} - TM number {id + 1}')

    return res_dict


def plot_2d_gt_vs_rec_histogram(samples,
                                operator_dict,
                                diagnostics_path,
                                response_dict,
                                reference_dict,
                                base_filename=None,
                                response=True,
                                relative=False,
                                type='single',
                                offset=0.,
                                plot_kwargs=None):
    """
    Plots the 2D histogram of reconstruction vs. ground-truth in either
    the data_space (if response_func = response) or the signal space (if
    response=False).
    If relative = True the relative error of the reconstruction is plotted
    vs. the ground-truth.
    It is possible to either plot the 2D histograms for reconstruction mean
    or instead the sample-averaged histograms (type=sampled).


    Parameters
    ----------
    samples: nifty8.re.evi.Samples
        nifty8.re.evi.Samples object containing the posterior samples of the
        reconstruction.
    operator_dict: dict
        Dictionary of operators for which the histogram should be plotted.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    response_dict: dict
        Dictionary containing the instrument response functions.
    reference_dict: dict, None
        Dictionary of reference arrays (e.g. ground-truth) to calculate the NWR.
    base_filename: str, None
        Base string of file name saved.
        If None, the plot is displayed and not saved.
    response: bool, True
        If True, the histogram for reconstruction vs. ground-truth is plotted.
    type: str, 'single'
        Either 'single' (default) taking the 2d histogram of the mean or
        'sampled' to get the sample averaged histogram.
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
        shape = exp(operator_dict[tuple(operator_dict)[0]](
            jft.mean(samples))).shape
        reshape = lambda x: np.tile(x, (shape[0], 1, 1, 1))
        R = lambda x: jft.Vector(
            {k: response_dict['mask_adj'](response_dict['mask'](reshape(x)))[0]
             for k in range(shape[0])})

    # TODO: unify cases
    if 'kernel' in response_dict:
        k = response_dict['kernel']
        Rs_sample_dict = {key: [R(op(s), k) for s in samples] for key, op in
                          operator_dict.items()}
        Rs_reference_dict = {key: R(ref, k) for key, ref in
                             reference_dict.items()}
    else:
        Rs_sample_dict = {key: [R(op(s)) for s in samples] for key, op in
                          operator_dict.items()}
        Rs_reference_dict = {key: R(ref) for key, ref in reference_dict.items()}

    for key in operator_dict.keys():
        res_list = []
        for Rs_sample in Rs_sample_dict[key]:
            for i, data_key in enumerate(Rs_sample.tree.keys()):
                if relative:
                    ref = Rs_reference_dict[key][data_key][
                        Rs_reference_dict[key][data_key] != 0]
                    samp = Rs_sample[data_key][
                        Rs_reference_dict[key][data_key] != 0]
                    res = np.abs(ref - samp) / ref
                else:
                    res = Rs_sample[data_key]
                    ref = Rs_reference_dict[key][data_key]
                if i == 0:
                    stacked_res = res.flatten()
                    stacked_ref = ref.flatten()
                else:
                    stacked_res = np.concatenate(
                        [stacked_res, res.flatten()]).flatten()
                    stacked_ref = np.concatenate(
                        [stacked_ref, ref.flatten()]).flatten()
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
                res_1d_array_list[i] = (np.abs(ref_list[i] - sample) /
                                        ref_list[i])
        if base_filename is not None:
            output_path = join(diagnostics_path,
                               f'{base_filename}hist_{key}.png')
        else:
            output_path = None
        plot_sample_averaged_log_2d_histogram(x_array_list=ref_list,
                                              y_array_list=res_1d_array_list,
                                              output_path=output_path,
                                              offset=offset,
                                              **plot_kwargs)
