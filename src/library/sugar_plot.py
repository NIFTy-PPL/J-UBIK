from matplotlib.colors import LogNorm
from os.path import join
import numpy as np
from jax import random, linear_transpose
import nifty8.re as jft
import nifty8 as ift

from .plot import plot_slices, plot_result
from .utils import get_data_domain, get_stats, create_output_directory, get_config
from ..library.chandra_observation import ChandraObservationInformation
from ..library.response import build_erosita_response_from_config
from ..library.sky_models import SkyModel

from .mf_plot import plot_rgb

"""FIXME: Clean this module - Plotting routines that are specific and are loading skymodels 
or reponses should go here"""


def plot_fused_data(obs_info, img_cfg, obslist, outroot, center=None):
    grid = img_cfg["grid"]
    data_domain = get_data_domain(grid)
    data = []
    for obsnr in obslist:
        info = ChandraObservationInformation(obs_info["obs" + obsnr], **grid, center=center)
        data.append(info.get_data(f"./data_{obsnr}.fits"))
    full_data = sum(data)
    full_data_field = ift.makeField(data_domain, full_data)
    plot_slices(full_data_field, outroot + "_full_data.png")


def plot_rgb_image(file_name_in, file_name_out, log_scale=False):
    import astropy.io.fits as pyfits
    from astropy.visualization import make_lupton_rgb
    import matplotlib.pyplot as plt
    color_dict = {0: "red", 1: "green", 2: "blue"}
    file_dict = {}
    for key in color_dict:
        file_dict[color_dict[key]] = pyfits.open(f"{file_name_in}_{color_dict[key]}.fits")[0].data
    rgb_default = make_lupton_rgb(file_dict["red"], file_dict["green"], file_dict["blue"],
                                  filename=file_name_out)
    if log_scale:
        plt.imshow(rgb_default, norm=LogNorm(), origin='lower')
    else:
        plt.imshow(rgb_default, origin='lower')


def plot_sample_and_stats(output_directory, operators_dict, sample_list, iteration=None,
                          log_scale=True, colorbar=True, dpi=300, plotting_kwargs=None,
                          rgb_min_sat=None, rgb_max_sat=None, plot_samples=True):
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
                                                           "samples", f"iteration_{iteration}"))
        if e_length == 3:
            rgb_result_path_samples = create_output_directory(join(results_path, "rgb", "samples",
                                                                   f"iteration_{iteration}"))
            rgb_result_path_stats = create_output_directory(join(results_path, "rgb", "stats"))

        filename_mean = join(stats_result_path, f"mean_it_{iteration}.png")
        filename_std = join(stats_result_path, f"std_it_{iteration}.png")

        # Plot samples
        # FIXME: works only for 2D outputs, add target capabilities

        if plot_samples:
            for i in range(n_samples):
                filename_samples = join(samples_result_path, f"sample_{i+1}_it_{iteration}.png")
                title = [f"Energy {ii+1}" for ii in range(e_length)]
                plotting_kwargs.update({'title': title})
                plot_result(operator_samples[i], output_file=filename_samples, logscale=log_scale,
                            colorbar=colorbar, dpi=dpi, adjust_figsize=True, **plotting_kwargs)
                rgb_name = join(results_path, f"rgb_{iteration}")

                # TODO this only works for 3 E-Bins
                if e_length == 3:
                    # Plot RGB
                    rgb_filename = join(rgb_result_path_samples, f"sample_{i+1}_{iteration}_rgb")
                    # sat_max = [rgb_max_sat[j] * operator_samples[i][j].max() for j in range(3)]
                    plot_rgb(operator_samples[i], rgb_filename, sat_min=rgb_min_sat,
                             sat_max=rgb_max_sat)
                    plot_rgb(operator_samples[i], rgb_filename+"_log", sat_min=rgb_min_sat,
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
            title = [f"Posterior mean (energy {ii+1})" for ii in range(e_length)]
            plot_result(mean, output_file=filename_mean, logscale=log_scale,
                        colorbar=colorbar, title=title, dpi=dpi, **plotting_kwargs)
            title = [f"Posterior std (energy {ii+1})" for ii in range(e_length)]
            plot_result(std, output_file=filename_std, logscale=log_scale,
                        colorbar=colorbar, title=title, dpi=dpi, **plotting_kwargs)

            if e_length == 3:
                rgb_name = join(rgb_result_path_stats, f"_mean_it_{iteration}_rgb")
                plot_rgb(mean, rgb_name, sat_min=rgb_min_sat,
                         sat_max=rgb_max_sat)
                plot_rgb(mean, rgb_name+"_log", sat_min=rgb_min_sat,
                         sat_max=None, log=True)


def plot_erosita_priors(key, n_samples, config_path, priors_dir, signal_response=False,
                        plotting_kwargs=None, common_colorbar=False, log_scale=True,
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
                        title=[f'E_min={emin}, E_max={emax}' for emin, emax in zip(e_min, e_max)],
                        common_colorbar=common_colorbar, **plotting_kwargs)

    if signal_response:  # FIXME: when R will be pickled, load the response from file
        tm_ids = cfg['telescope']['tm_ids']
        n_modules = len(tm_ids)

        spix = cfg['grid']['sdim']
        epix = cfg['grid']['edim']
        response_dict = build_erosita_response_from_config(config_path)

        mask_adj = linear_transpose(response_dict['mask'],
                                        np.zeros((n_modules, epix, spix, spix)))

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
                    plot_result(samps, output_file=filename.format(key), logscale=log_scale,
                                title=[f'E_min={emin}, E_max={emax}' for emin, emax in
                                       zip(e_min, e_max)],
                                common_colorbar=common_colorbar, adjust_figsize=adjust_figsize)
