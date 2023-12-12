import jax
import numpy as np
import pickle
from os.path import join
from functools import partial
from jax.tree_util import tree_map
from jax import numpy as jnp

import nifty8 as ift
import nifty8.re as jft

from .utils import get_config, create_output_directory
from .plot import plot_result, plot_sample_averaged_log_2d_histogram, plot_histograms
from .sky_models import create_sky_model_from_config
from .response import build_callable_from_exposure_file


def _build_full_mask_func_from_exp_files(exposure_file_names,
                                         threshold=None):
    """ Buils the full mask function for the reconstruction given the exposure_file_names

    Parameters
    ----------
    exposure_file_names : list[str]
        A list of filenames of exposure files to load.
        Files should be in a .npy or .fits format.
    threshold: float, None
        A threshold value below which flags are set to zero (e.g., an exposure cut).
        If None (default), no threshold is applied.

    Returns
    -------
        function: A callable that applies a mask to an input array (e.g. an input sky) and returns
        a `nifty8.re.Vector` containing a dictionary of read-out inputs.
    """

    def _set_zero(x, exp):
        try:
            x.at[exp == 0].set(0)
        except:
            x[exp == 0] = 0
        return jft.Vector({0: x})

    def _builder(exposures, trsh):
        if threshold < 0:
            raise ValueError("threshold should be positive!")
        if threshold is not None:
            exposures[exposures < threshold] = 0
        summed_exp = np.sum(exposures, axis=0)
        return partial(_set_zero, exp=summed_exp)

    return build_callable_from_exposure_file(_builder, exposure_file_names, trsh=threshold)


def get_diagnostics_from_file(diagnostic_builder,
                              diagnostics_path,
                              minimization_output_path,
                              config_path,
                              output_operators_keys,
                              **kwargs):
    """ Creates sample list and sky model and passes the according sample list
    to the considered diagnostic builder function.

    Parameters
    ----------
    diagnostic_builder: callable
        Diagnostic function to be called on the build sample list.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    minimization_output_path: str
        Path base to the minimization output file containing the samples and
        the optimizer state.
    config_path: str
        Path to config file
    output_operators_keys: list
        List of strings for keys of operators that shall be considered in the
        diagnostic_builder function.
    """

    # Build diagnostics directory
    create_output_directory(diagnostics_path)

    # Get config info
    cfg = get_config(config_path)
    grid_info = cfg['grid']

    # Create sky operators
    sky_dict = create_sky_model_from_config(config_path)
    sky_dict.pop('pspec')

    # Load position space sample list
    with open(minimization_output_path, "rb") as file:
        samples, _ = pickle.load(file)

    samples = samples.samples
    padding_diff = int((grid_info['npix'] - round(grid_info['npix']/grid_info['padding_ratio']))/2)
    pos_sp_sample_dict = {key: [jax.vmap(op)(samples)[i][padding_diff:-padding_diff,
                                padding_diff:-padding_diff] for i in
                                range(len(jax.vmap(op)(samples)))]
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
    Computes uncertainty-weighted residuals (UWRs) given the position space sample list and
    plots them and the according histogram. Definition:
    :math:`uwr = \\frac{s-gt}{\\sigma_{s}}`,
    where `s` is the signal, `gt` is the ground truth,
    and `sigma_{s}` is the standard deviation of `s`.

    Parameters
    ----------
    pos_sp_sample_dict: dict
        Dictionary of position space sample lists for the operator specified by the key.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    cfg: dict
        Reconstruction config dictionary.
    reference_dict: dict, None
        Dictionary of reference arrays (e.g. groundtruth for mock) to calculate the UWR. If the
        reference_dict is set to none we assume the reference to be zero and the measure defaults
        to the uncertainty-weighted mean (UWM)
    output_dir_base: str, None
        Base string of file name saved.
    mask: bool, False
        If true the outout is masked by a full mask generated from all exposure files.
    abs: bool, False
        If true the absolute value of the residual is calculated and plotted.
    n_bins: int, None
        Number of bins in the histogram plotted. If None, no histogram is plotted.
    range: tuple, None
        Range of the histogram. If None, the range defaults to (-5, 5)
    log: bool, True
        If True, the logrithm of the samples and reference are considered.
    plot_kwargs: dict, None
        Dictionary of plotting arguments for plot_results

    Returns:
    ----------
    diag_dict: dict
        Dictionary of UWRs or UWMs for each operator specified in the input
        pos_sp_sample_dict.
    """

    mpi_master = ift.utilities.get_MPI_params()[3]
    diag_dict = {}
    if log:
        pos_sp_sample_dict = {key: np.log(value) for key, value in pos_sp_sample_dict.items()}
        if reference_dict is not None:
            reference_dict = {key: np.log(value) for key, value in reference_dict.items()}
    for key, pos_sp_sl in pos_sp_sample_dict.items():
        mean = np.stack(pos_sp_sl).mean(axis=0)
        std = np.stack(pos_sp_sl).std(axis=0, ddof=1)

        if reference_dict is None:
            print("No reference ground truth provided. "
                "Uncertainty weighted residuals calculation defaults to uncertainty-weighted mean.")
            diag = mean / std

        else:
            if abs:
                diag = (mean - reference_dict[key]).abs() / std
            else:
                diag = (reference_dict[key]-mean) / std

        if mask:
            tel_info = cfg['telescope']
            file_info = cfg['files']
            exposure_file_names = [join(file_info['obs_path'], f'{key}_' + file_info['exposure'])
                                  for key in tel_info['tm_ids']]
            full_mask_func = _build_full_mask_func_from_exp_files(exposure_file_names,
                                                                  tel_info['exp_cut'])
            diag = full_mask_func(diag)
        diag_dict[key] = diag.tree[0]

        if mpi_master and output_dir_base is not None:
            with open(join(diagnostics_path, f'{output_dir_base}{key}.pkl'), 'wb') as file:
                pickle.dump(diag_dict[key], file)
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
            plot_result(diag_dict[key], output_file=join(diagnostics_path,
                                                         f'{output_dir_base}{key}.png'),
                        **plot_kwargs)

            if n_bins:
                if range is None:
                    range = (-5, 5)
                hist, edges = np.histogram(diag_dict[key].reshape(-1), bins=n_bins, range=range)
                title = plot_kwargs['title'] if plot_kwargs is not None else None
                plot_histograms(hist, edges,
                                join(diagnostics_path, f'{output_dir_base}hist_{key}.png'),
                                title=title)
    return diag_dict


def compute_noise_weighted_residuals(pos_sp_sample_dict,
                                     diagnostics_path,
                                     cfg,
                                     response_dict,
                                     reference_dict,
                                     output_dir_base=None,
                                     min_counts=0,
                                     abs=False,
                                     n_bins=None,
                                     extent=None,
                                     plot_kwargs=None):
    """
    Computes noise-weighted residuals (NWRs) given the position space sample list and
    plots the according histogram. Definition:
    :math:`nwr = \\frac{Rs-d}{\\sqrt_{Rs}}`,
    where `s` is the signal, 'R' is the response and  `d` is the reference data.

    Parameters
    ----------
    pos_sp_sample_dict: dict
        Dictionary of position space sample lists for the operator specified by the key.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    cfg: dict
        Reconstruction config dictionary
    response_dict: dict
        Dictionary with response callables.
    reference_dict: dict, None
        Dictionary of reference arrays (e.g. data) to calculate the NWR.
    output_dir_base: str, None
        Base string of file name saved.
    abs: bool, False
        If true the absolute value of the residual is calculated and plotted.
    n_bins: int, None
        Number of bins in the histogram plotted. If None, no histogram is plotted.
    min_counts: `int`, minimum number of data counts for which the residuals will be calculated.
    extent: tuple, None
        Range of the histogram. If None, the range defaults to (-5, 5)
    plot_kwargs: dict, None
        Dictionary of plotting keyword arguments for plot_result.
    """
    mpi_master = ift.utilities.get_MPI_params()[3]
    mask_adj = jax.linear_transpose(response_dict['mask'],
                                    np.zeros((2, 426, 426))) # FIXME: remove hardcoded value
    mask_adj = jax.vmap(mask_adj, in_axes=0, out_axes=1)

    # R = lambda x: mask_adj(response_dict['R'](x))
    R = response_dict['R']
    R = jax.vmap(R, in_axes=0, out_axes=1)

    nwr_func = lambda x, y: (x - y) / jnp.sqrt(x)

    nwr_func = jax.vmap(nwr_func, in_axes=0, out_axes=1)

    if min_counts > 0:
        low_count_loc = lambda x: x < min_counts
        low_count_loc = tree_map(low_count_loc, reference_dict)
        mask_low_count = lambda x, cond: jnp.where(cond, np.nan, x)

    nwr_dict = {}
    unmasked_nwr_dict = {}
    mean_hist_dict = {}
    edges_dict = {}
    for key, sample_list in pos_sp_sample_dict.items():
        sample_list = np.array(sample_list)
        Rs_sample = R(sample_list)
        nwr = tree_map(lambda x, y: nwr_func(x, y), Rs_sample, reference_dict)
        if abs:
            nwr = tree_map(np.abs, nwr)
        if min_counts > 0:
            nwr = tree_map(mask_low_count, nwr, low_count_loc)
        nwr_dict[key] = nwr
        unmasked_nwr_dict[key] = mask_adj(nwr)
        if mpi_master:
            if plot_kwargs is None:
                plot_kwargs = {}
            if 'cmap' not in plot_kwargs:
                plot_kwargs.update({'cmap': 'seismic'})
            if 'vmin' not in plot_kwargs:
                plot_kwargs.update({'vmin': -5})
            if 'vmax' not in plot_kwargs:
                plot_kwargs.update({'vmax': 5})

            tm_ids = tuple(nwr.tree.keys())
            for i, id in enumerate(tm_ids):
                res_path = create_output_directory(join(diagnostics_path, f'tm{id}/{key}/'))
                if 'title' not in plot_kwargs:
                    plot_kwargs.update({'title': f'{key} NWR - TM{id}'})
                plot_result(unmasked_nwr_dict[key][0][i,:,:,:],
                            output_file=join(res_path,
                                             f'{output_dir_base}{key}_tm{id}_samples.png'),
                            adjust_figsize=True,
                            **plot_kwargs,
                            )

                plot_result(np.mean(unmasked_nwr_dict[key][0][i,:,:,:], axis=0),
                            output_file=join(res_path,
                                             f'{output_dir_base}{key}_tm{id}_mean.png'),
                            adjust_figsize=True,
                            **plot_kwargs,
                            )

        if n_bins is not None:
            if extent is None:
                extent = (-5, 5)
            hist_func = lambda x: jnp.histogram(x.reshape(-1), bins=n_bins, range=extent)[0]
            edges_func = lambda x: jnp.histogram(x.reshape(-1), bins=n_bins, range=extent)[1]
            hist = tree_map(jax.vmap(hist_func, in_axes=0, out_axes=0), nwr)
            edges = tree_map(edges_func, nwr)
            mean_hist = tree_map(lambda x: np.mean(x, axis=0), hist)
            mean_hist_dict[key] = mean_hist
            edges_dict[key] = edges

            for id in mean_hist.tree.keys():
                if mpi_master:
                    plot_histograms(mean_hist[id], edges[id], join(diagnostics_path,
                                                                   f'{output_dir_base}hist_{key}_tm{id}.png'),
                                    logy=False, title=f'NWR mean {key} - TM{id}')
    if mpi_master:
        return nwr_dict, unmasked_nwr_dict, mean_hist_dict, edges_dict




def plot_2d_gt_vs_rec_histogram(pos_sp_sample_dict,
                                diagnostics_path,
                                cfg,
                                response_func,
                                reference_dict,
                                output_dir_base,
                                type= 'single',
                                relative=False,
                                plot_kwargs=None):
    """
    Plots the 2D histogram of reconstruction vs. ground-truth in either
    the data_space (if response_func = response) or the signal space (if response_func= None).
    If relative = True the realtive error of the reconstruction is plotted vs. the ground-truth.
    It is possible to either plot the 2D histograms for reconstruction mean or instead the
    sample-averaged histograms (type=sampled)


    Parameters
    ----------
    pos_sp_sample_dict: dict
        Dictionary of position space sample lists for the operator specified by the key.
    diagnostics_path: str
        Path to the reconstruction diagnostic files.
    cfg: dict
        Recosntruction config dictionary.
    response_func: callable
        Callable response function that can be applied to the signal returning an object of the type
        'jft.Vector', that conatins the according data space representations.
    reference_dict: dict, None
        Dictionary of reference arrays (e.g. ground-truth) to calculate the NWR.
    output_dir_base: str, None
        Base string of file name saved.
    type: str, 'single'
        Either 'single' (default) taking the 2d histogram of the mean or 'sampled' to get
        the sample averaged histogram.
    relative: bool. False
        If False, the histogram for reconstruction vs. ground-truth is plotted. If True,
        the relative error vs. ground-truth histogram is generated.
    plot_kwargs: dict, None
        Dictionary of plotting arguments
    """
    mpi_master = ift.utilities.get_MPI_params()[3]
    if response_func is None:
        tel_info = cfg['telescope']
        file_info = cfg['files']
        exposure_file_names = [join(file_info['obs_path'], f'{key}_' + file_info['exposure'])
                               for key in tel_info['tm_ids']]
        response_func = _build_full_mask_func_from_exp_files(exposure_file_names,
                                                             tel_info['exp_cut'])
    Rs_sample_dict = {key: [response_func(sample) for sample in sample_list]
                      for key, sample_list in pos_sp_sample_dict.items()}
    Rs_reference_dict = {key: response_func(ref) for key, ref in reference_dict.items()}

    for key, sl in pos_sp_sample_dict.items():
        res_list = []
        for Rs_sample in Rs_sample_dict[key]:
            for i, data_key in enumerate(Rs_sample.tree.keys()):
                if relative:
                    ref = Rs_reference_dict[key][data_key][Rs_reference_dict[key][data_key] != 0]
                    samp = Rs_sample[data_key][Rs_reference_dict[key][data_key] != 0]
                    res = np.abs(ref-samp)/ref
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
        ref_list = len(res_1d_array_list)*[stacked_ref]
        if relative:
            for i, sample in enumerate(res_1d_array_list):
                res_1d_array_list[i] = np.abs(ref_list[i] - sample) / ref_list[i]
        if mpi_master and output_dir_base:
            output_path = join(diagnostics_path, f'{output_dir_base}hist_{key}.png')
            plot_sample_averaged_log_2d_histogram(x_array_list=ref_list,
                                                  y_array_list=res_1d_array_list,
                                                  output_path=output_path,
                                                  **plot_kwargs)

