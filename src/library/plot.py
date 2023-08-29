import os
import math

import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .utils import get_data_domain, get_config, create_output_directory
from ..library.sky_models import SkyModel
from ..library.chandra_observation import ChandraObservationInformation


def plot_result(array, domain=None, output_file=None, logscale=False, title=None, colorbar=True,
                figsize=(5,5), dpi=100, cbar_formatter=None, **kwargs):
    """
    Plot a 2D array using imshow() from the matplotlib library.

    Parameters:
    -----------
    array : numpy.ndarray
        The 2D array to plot.
    domain : dict, optional
        The domain of the array.
    output_file : str, optional
        The name of the file to save the plot to.
    logscale : bool, optional
        Whether to use a logarithmic scale for the color map.
    title : str, optional
        The title of the plot.
    colorbar : bool, optional
        Whether to show the color bar.
    figsize : tuple, optional
        The size of the figure in inches.
    dpi : int, optional
        The resolution of the figure in dots per inch.
    cbar_formatter : matplotlib.ticker.Formatter, optional
        The formatter for the color bar ticks.
    kwargs : dict, optional
        Additional keyword arguments to pass to imshow().

    Returns:
    --------
    None
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pltargs = {"origin": "lower", "cmap": "viridis"}
    if domain is not None:
        half_fov = domain["distances"][0] * domain["shape"][
            0] / 2.0 / 60  # conv to arcmin FIXME: works only for square array
        pltargs["extent"] = [-half_fov, half_fov] * 2
        ax.set_xlabel("FOV [arcmin]")
        ax.set_ylabel("FOV [arcmin]")
    if logscale:
        pltargs["norm"] = LogNorm()
    pltargs.update(**kwargs)
    im = ax.imshow(array, **pltargs)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, format=cbar_formatter)
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_file}.")
    else:
        plt.show()
    plt.close()


def plot_slices(field, outname, logscale=False):
    img = field.val
    npix_e = field.domain.shape[-1]
    nax = np.ceil(np.sqrt(npix_e)).astype(int)
    half_fov = field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0 / 60. # conv to arcmin
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-half_fov, half_fov] * 2}
    if logscale == True:
        pltargs["norm"] = LogNorm()

    fig, ax = plt.subplots(
        nax, nax, figsize=(11.7, 8.3), sharex=True, sharey=True, dpi=200
    )
    ax = ax.flatten()
    for ii in range(npix_e):
        im = ax[ii].imshow(img[:, :, ii], **pltargs)
        cb = fig.colorbar(im, ax=ax[ii])
    fig.tight_layout()
    if outname != None:
        fig.savefig(outname)
    plt.close()


def plot_results(field_list, title_list, outname, logscale=False, ncols=3, nrows=1, cbar_shrink=1.0, pltargs_list=None):
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, dpi=300, figsize=(11.7, 8.3))
    ax = ax.ravel()
    for i, field in enumerate(field_list):
        img = field.val
        half_fov = field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0 / 60  # conv to arcmin
        pltargs = {"origin": "lower", "cmap": "viridis", "extent": [-half_fov, half_fov] * 2}
        if logscale == True:
            pltargs["norm"] = LogNorm()
        pltargs.update(**pltargs_list[i])
        im = ax[i].imshow(img, **pltargs)
        cb = fig.colorbar(im, ax=ax[i], shrink=cbar_shrink)
        ax[i].set_axis_off()  # remove axis lines and labels
        ax[i].set_title(title_list[i])
        fig.tight_layout()  # center the images
    if outname != None:
        fig.savefig(outname, bbox_inches='tight', pad_inches=0)
    plt.close()


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


def plot_image_from_fits(file_name_in, file_name_out, log_scale=False):
    import matplotlib.pyplot as plt
    from astropy.utils.data import get_pkg_data_filename
    from astropy.io import fits
    image_file = get_pkg_data_filename(file_name_in)
    image_data = fits.getdata(image_file, ext=0)
    plt.figure()
    plt.imshow(image_data, norm=LogNorm())
    plt.savefig(file_name_out)


def plot_single_psf(psf, outname, logscale=True, vmin=None, vmax=None):
    half_fov = psf.domain[0].distances[0] * psf.domain[0].shape[0] / 2.0 / 60  # conv to arcmin
    psf = psf.val  # .reshape([1024, 1024])
    pltargs = {"origin": "lower", "cmap": "cividis", "extent": [-half_fov, half_fov] * 2}
    if logscale == True:
        pltargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots()
    psf_plot = ax.imshow(psf, **pltargs)
    fig.colorbar(psf_plot)
    fig.tight_layout()
    fig.savefig(outname, dpi=1500)
    plt.close()


def plot_psfset(fname, outname, npix, n, in_one=True):
    fileloader = np.load(fname, allow_pickle=True).item()
    psf = fileloader["psf_sim"]
    if in_one:
        psfset = psf[0]
        for i in range(1, n ** 2):
            psfset = psfset + psf[i]
        plot_single_psf(psfset, outname + "psfset.png", logscale=True)

    else:
        p = ift.Plot()
        for k in range(10):
            p.add(psf[k], title=f"{k}", norm=LogNorm())
        p.output(name=outname + "psfs.png", xsize=20, ysize=20)


def _append_key(s, key):
    if key == "":
        return s
    return f"{s} ({key})"


def _plot_samples(filename, samples, plotting_kwargs):
    samples = list(samples)

    if isinstance(samples[0].domain, ift.DomainTuple):
        samples = [ift.MultiField.from_dict({"": ss}) for ss in samples]
        # if ground_truth is not None:
        #     ground_truth = ift.MultiField.from_dict({"": ground_truth})
    if not all(isinstance(ss, ift.MultiField) for ss in samples):
        raise TypeError
    keys = samples[0].keys()

    p = ift.Plot()
    for kk in keys:
        single_samples = [ss[kk] for ss in samples]

        if ift.plot.plottable2D(samples[0][kk]):
            # if ground_truth is not None:
            # p.add(ground_truth[kk], title=_append_key("Ground truth", kk),
            #       **plotting_kwargs)
            # p.add(None)
            for ii, ss in enumerate(single_samples):
                # if (ground_truth is None and ii == 16) or (ground_truth is not None and ii == 14):
                #     break
                p.add(ss, title=_append_key(f"Sample {ii}", kk), **plotting_kwargs)
        else:
            n = len(samples)
            alpha = n * [0.5]
            color = n * ["maroon"]
            label = None
            # if ground_truth is not None:
            #     single_samples = [ground_truth[kk]] + single_samples
            #     alpha = [1.] + alpha
            #     color = ["green"] + color
            #     label = ["Ground truth", "Samples"] + (n-1)*[None]
            p.add(single_samples, color=color, alpha=alpha, label=label,
                  title=_append_key("Samples", kk), **plotting_kwargs)
    p.output(name=filename)


def _plot_stats(filename, op, sl, plotting_kwargs):
    try:
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    mean, var = sl.sample_stat(op)
    p = ift.Plot()
    # if op is not None: TODO: add Ground Truth plotting capabilities
    #     p.add(op, title="Ground truth", **plotting_kwargs)
    p.add(mean, title="Mean", **plotting_kwargs)
    p.add(var.sqrt(), title="Standard deviation", **plotting_kwargs)
    p.output(name=filename, ny=2)
    # print("Output saved as {}.".format(filename))


def plot_sample_and_stats(output_directory, operators_dict, sample_list, iterator, plotting_kwargs):
    for key in operators_dict:
        op = operators_dict[key]
        results_path = os.path.join(output_directory, key)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        filename = os.path.join(output_directory, key, "stats_{}.png".format(iterator))
        filename_samples = os.path.join(output_directory, key, "samples_{}.png".format(iterator))

        _plot_stats(filename, op, sample_list, plotting_kwargs)
        _plot_samples(filename_samples, sample_list.iterator(op), plotting_kwargs)


def plot_energy_slices(field, file_name, title=None, plot_kwargs={}):
    """
    Plots the slices of a 3-dimensional field along the energy dimension.

    Parameters:
    ----------
    field : ift.Field
        The field to plot.
    file_name : str
        The name of the file to save the plot.
    title : str or None
        The title of the plot. Default is None.
    plot_kwargs : `dict` keyword arguments for plotting.
        If True, the plot uses a logarithmic scale. Default is False.

    Raises:
    -------
    ValueError : if the domain of the field is not as expected.

    Returns:
    --------
    None
    """
    domain = field.domain
    if not isinstance(domain, ift.DomainTuple) or len(domain[0].shape) !=2:
        raise ValueError(f"Expected DomainTuple with the first space"
                         f"being a 2-dim RGSpace, but got {domain}")

    if len(domain) == 2 and len(domain[1].shape) != 1:
        raise ValueError(f"Expected DomainTuple with the second space"
                         f"being a 1-dim RGSpace, but got {domain}")

    if len(domain) == 1:
        p = ift.Plot()
        p.add(field, **plot_kwargs)
        p.output(name=file_name)

    elif len(domain) == 2:
        p = ift.Plot()
        for i in range(field.shape[2]):
            slice = ift.Field(ift.DomainTuple.make(domain[0]), field.val[:, :, i])
            p.add(slice, title=f'{title}_e_bin={i}', **plot_kwargs)
        p.output(name=file_name)
    else:
        raise NotImplementedError


def plot_energy_slice_overview(field_list, field_name_list, file_name, title=None, logscale=False):
    """
    Plots a list of fields in one plot separated by energy bins

    Parameters:
    ----------
    field_list : List of ift.Fields
                 The field to plot.
    file_name : str
        The name of the file to save the plot.
    title : str or None
        The title of the plot. Default is None.
    logscale : bool
        If True, the plot uses a logarithmic scale. Default is False.

    Raises:
    -------
    ValueError : if the domain of the field is not as expected.
    ValueError: If the number of field names does not match the number of fields.

    Returns:
    --------
    None
    """
    domain = field_list[0].domain
    if any(field.domain != domain for field in field_list):
        raise ValueError('All fields need to have the same domain.')

    if not isinstance(domain, ift.DomainTuple) or len(domain[0].shape) != 2:
        raise ValueError(f"Expected DomainTuple with the first space "
                         f"being a 2-dim RGSpace, but got {domain}")

    if len(domain) == 2 and len(domain[1].shape) != 1:
        raise ValueError(f"Expected DomainTuple with the second space "
                         f"being a 1-dim RGSpace, but got {domain}")

    if len(field_list) != len(field_name_list):
        raise ValueError("Every field needs a name")

    pltargs = {"origin": "lower", "cmap": "cividis"}
    if logscale:
        pltargs["norm"] = LogNorm()
    cols = math.ceil(math.sqrt(len(field_list)))  # Calculate number of columns
    rows = math.ceil(len(field_list) / cols)
    if len(domain) == 1:
        if len(field_list) == 1:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11.7, 8.3),
                                   sharex=True, sharey=True, dpi=200)
            im = ax.imshow(field_list[0].val, **pltargs)
            ax.set_title(f'{title}_{field_name_list[0]}')
        else:
            fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(11.7, 8.3),
                                   sharex=True, sharey=True, dpi=200)
            ax = ax.flatten()
            for i, field in enumerate(field_list):
                im = ax[i].imshow(field.val, **pltargs)
                ax[i].set_title(f'{title}_{field_name_list[i]}')
        fig.tight_layout()
        fig.savefig(f'{file_name}')
        plt.close()
    elif len(domain) == 2:
        for i in range(domain[1].shape[0]):
            if len(field_list) == 1:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11.7, 8.3),
                                       sharex=True, sharey=True, dpi=200)
                im = ax.imshow(field_list[0].val, **pltargs)
                ax.set_title(f'{title}_{field_name_list[0]}')
            else:
                fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(11.7, 8.3),
                                       sharex=True, sharey=True, dpi=200)
                ax = ax.flatten()
                for j, field in enumerate(field_list):
                    im = ax[j].imshow(field.val[:, :, i], **pltargs)
                    ax[j].set_title(f'{field_name_list[j]}')
            fig.tight_layout()
            fig.savefig(f'{file_name}_e_bin={i}.png')
            plt.close()
    else:
        raise NotImplementedError


def plot_erosita_priors(n_samples, config_path, response_path, priors_dir,
                        plotting_kwargs=None, common_colorbar=False):
    priors_dir = create_output_directory(priors_dir)
    cfg = get_config(config_path)  # load config

    if plotting_kwargs is None:
        plotting_kwargs = {}

    if 'norm' in plotting_kwargs:
        norm = plotting_kwargs.pop('norm')
        norm = type(norm)
    else:
        norm = None

    sky_dict = SkyModel(config_path).create_sky_model()
    plottable_ops = sky_dict.copy()

    positions = []
    for sample in range(n_samples):
        positions.append(ift.from_random(plottable_ops['sky'].domain))

    plottable_samples = plottable_ops.copy()
    for key, val in plottable_samples.items():
        plottable_samples[key] = [val.force(pos) for pos in positions]

    filename_base = priors_dir + 'priors_{}.png'
    _plot_erosita_samples(common_colorbar, n_samples, norm, plottable_samples,
                          plotting_kwargs, filename_base, ' prior ', 'Prior samples')

    if response_path is not None:  # FIXME: when R will be pickled, load from file
        tm_ids = cfg['telescope']['tm_ids']
        plottable_ops.pop('pspec')

        resp_dict = load_erosita_response(config_path, priors_dir)  # FIXME

        for tm_id in tm_ids:
            tm_key = f'tm_{tm_id}'
            R = resp_dict[tm_key]['mask'].adjoint @ resp_dict[tm_key]['R']
            plottable_samples = {}

            for key, val in plottable_ops.items():
                SR = R @ val
                plottable_samples[key] = [SR.force(pos) for pos in positions]

            res_path = priors_dir + f'tm{tm_id}/'
            filename = res_path + f'sr_tm_{tm_id}_priors'
            filename += '_{}.png'
            _plot_erosita_samples(common_colorbar, n_samples, norm, plottable_samples,
                                  plotting_kwargs, filename, f'tm {tm_id} prior signal response ',
                                  'Signal response')


def _plot_erosita_samples(common_colorbar, n_samples, norm, plottable_samples,
                          plotting_kwargs, filename_base='priors/samples_{}.png',
                          title_base='', log_base=None):
    for key, val in plottable_samples.items():
        if common_colorbar:
            vmin = min(np.min(val[i].val) for i in range(n_samples))
            vmax = max(np.max(val[i].val) for i in range(n_samples))
            if float(vmin) == 0.:
                vmin = 1e-18  # to prevent LogNorm throwing errors
        else:
            vmin = vmax = None
        p = ift.Plot()
        for i in range(n_samples):
            if norm is not None:
                p.add(val[i], norm=norm(vmin=vmin, vmax=vmax),
                      title=title_base + key, **plotting_kwargs)
            else:
                p.add(val[i], vmin=vmin, vmax=vmax,
                      title=title_base + key, **plotting_kwargs)
            if 'title' in plotting_kwargs:
                del (plotting_kwargs['title'])
        filename = filename_base.format(key)
        p.output(name=filename, **plotting_kwargs)
        if log_base is None:
            log_base = 'Samples'
        print(f'{log_base} saved as {filename}.')


def plot_histograms(hist, edges, filename, logx=False, logy=False, title=None):
    plt.bar(edges[:-1], hist, width=edges[0]-edges[1])
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Histogram saved as {filename}.")
