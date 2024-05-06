import math
import nifty8 as ift
import nifty8.re as jft
import numpy as np
import jax
import matplotlib.pyplot as plt
from jax import random
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join


def plot_result(array, domains=None, output_file=None, logscale=False, title=None, colorbar=True,
                figsize=(8, 8), dpi=100, cbar_formatter=None, n_rows=None, n_cols=None,
                adjust_figsize=False, common_colorbar=False, share_x=True, share_y=True, **kwargs):
    """
    Plot a 2D array using imshow() from the matplotlib library.

    Parameters:
    -----------
    array : numpy.ndarray
        Array of images. The first index indices through the different images
        (e.g., shape = (5, 128, 128)).
    domains : list[dict], optional
        List of domains. Each domain should correspond to each image array.
    output_file : str, optional
        The name of the file to save the plot to.
    logscale : bool, optional
        Whether to use a logarithmic scale for the color map.
    title : list[str], optional
        The title of each individual plot in the array.
    colorbar : bool, optional
        Whether to show the color bar.
    figsize : tuple, optional
        The size of the figure in inches.
    dpi : int, optional
        The resolution of the figure in dots per inch.
    cbar_formatter : matplotlib.ticker.Formatter, optional
        The formatter for the color bar ticks.
    n_rows : int
        Number of columns of the final plot.
    n_cols : int, optional
        Number of rows of the final plot.
    adjust_figsize : bool, optional
        Whether to automatically adjust the size of the figure.
    common_colorbar : bool, optional
        Whether to use the same color bar for all images. Overrides vmin and vmax.
    share_x : bool, optional
        Whether to share the x axis.
    share_y : bool, optional
        Whether to share the y axis.
    kwargs : dict, optional
        Additional keyword arguments to pass to imshow().

    Returns:
    --------
    None
    """

    shape_len = array.shape
    if len(shape_len) < 2 or len(shape_len) > 3:
        raise ValueError("Wrong input shape for array plot!")
    if len(shape_len) == 2:
        array = array[np.newaxis, :, :]

    n_plots = array.shape[0]

    def _get_n_rows_from_n_samples(n_samples):
        """
        A function to get the number of rows from the given number of samples.

        Parameters:
        ----------
            n_samples: `int`. The number of samples.

        Returns:
        -------
            `int`: The number of rows.
        """
        threshold = 2
        n_rows = 1
        if n_samples == 2:
            return n_rows

        while True:
            if n_samples < threshold:
                return n_rows

            threshold = 4 * threshold + 1
            n_rows += 1

    if n_rows is None:
        n_rows = _get_n_rows_from_n_samples(n_plots)

    if n_cols is None:
        if n_plots % n_rows == 0:
            n_cols = n_plots // n_rows
        else:
            n_cols = n_plots // n_rows + 1

    if adjust_figsize:
        x = int(n_cols/n_rows)
        y = int(n_rows/n_cols)
        if x == 0:
            x = 1
        if y == 0:
            y = 1
        figsize = (x*figsize[0], y*figsize[1])

    n_ax = n_rows * n_cols
    n_del = n_ax - n_plots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, dpi=dpi, sharex=share_x,
                             sharey=share_y)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    pltargs = {"origin": "lower", "cmap": "viridis"}
    pltargs.update({'vmin': kwargs.get('vmin', None),
                    'vmax': kwargs.get('vmax', None)})

    for i in range(n_plots):
        if array[i].ndim != 2:
            raise ValueError("All arrays to plot must be 2-dimensional!")

        if domains is not None:
            half_fov = domains[i]["distances"][0] * domains[i]["shape"][
                0] / 2.0 / 60  # conv to arcmin FIXME: works only for square array
            pltargs["extent"] = [-half_fov, half_fov] * 2
            axes[i].set_xlabel("FOV [arcmin]")
            axes[i].set_ylabel("FOV [arcmin]")

        if "vmin" in kwargs:
            vmin = kwargs["vmin"]
        else:
            vmin = None
        if "vmax" in kwargs:
            vmax = kwargs["vmax"]
        else:
            vmax = None

        if colorbar and common_colorbar:
            vmin = min(np.min(array[i]) for i in range(n_plots))
            vmax = max(np.max(array[i]) for i in range(n_plots))

        if logscale:
            if vmin is not None and float(vmin) == 0.:
                vmin = 1e-18  # to prevent LogNorm throwing errors
            pltargs["norm"] = LogNorm(vmin, vmax)
        else:
            kwargs.update({'vmin': vmin, 'vmax': vmax})

        pltargs.update(**kwargs)
        im = axes[i].imshow(array[i], **pltargs)

        if title is not None:
            if isinstance(title, list):
                axes[i].set_title(title[i])
            else:
                fig.suptitle(title)

        if colorbar:
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax, format=cbar_formatter)
    for i in range(n_del):
        fig.delaxes(axes[n_plots+i])
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_file}.")
        plt.close()
    else:
        plt.show()


def plot_slices(field, outname, logscale=False):
    img = field.val
    npix_e = field.domain.shape[-1]
    nax = np.ceil(np.sqrt(npix_e)).astype(int)
    half_fov = field.domain[0].distances[0] * \
        field.domain[0].shape[0] / 2.0 / 60.  # conv to arcmin
    pltargs = {"origin": "lower", "cmap": "cividis",
               "extent": [-half_fov, half_fov] * 2}
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
    half_fov = psf.domain[0].distances[0] * \
        psf.domain[0].shape[0] / 2.0 / 60  # conv to arcmin
    psf = psf.val  # .reshape([1024, 1024])
    pltargs = {"origin": "lower", "cmap": "cividis",
               "extent": [-half_fov, half_fov] * 2}
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


def _get_n_rows_from_n_samples(n_samples):
    """
    A function to get the number of rows from the given number of samples.

    Parameters:
    ----------
        n_samples: `int`. The number of samples.

    Returns:
    -------
        `int`: The number of rows.
    """
    threshold = 2
    n_rows = 1
    if n_samples == 2:
        return n_rows

    while True:
        if n_samples < threshold:
            return n_rows

        threshold = 4*threshold + 1
        n_rows += 1


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
    if not isinstance(domain, ift.DomainTuple) or len(domain[0].shape) != 2:
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
            slice = ift.Field(ift.DomainTuple.make(
                domain[0]), field.val[:, :, i])
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


def plot_histograms(hist, edges, filename, logx=False, logy=False, title=None):
    plt.bar(edges[:-1], hist, width=edges[0] - edges[1])
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Histogram saved as {filename}.")


def plot_sample_averaged_log_2d_histogram(x_array_list, x_label, y_array_list, y_label,
                                          x_lim=None, y_lim=None, bins=100, dpi=400,
                                          title=None, output_path=None, offset=None, figsize=None):
    """ Plot a 2d histogram for the arrays given for x_array and y_array.


    Parameters:
    -----------
    x_array_list : list of numpy.ndarray
        list of samples of x_axis array of 2d-histogram
    x_label : string
        x-axis label of the 2d-histogram
    y_array_list : numpy.ndarray
        list of samples of y-axis array of 2d-histogram
    y_label : string
        y-axis label of the 2d-histogram
    bins : int
        Number of bins of the 2D-histogram
    dpi : int, optional
        Resolution of the figure
    title : string, optional
        Title of the 2D histogram
    output_path : string, optional
        Output directory for the plot. If None (Default) the plot is not saved.

    Returns:
    --------
    None
    """
    if len(x_array_list) != len(y_array_list):
        raise ValueError('Need same number of samples for x- and y-axis.')

    # Add small offset to avoid logarithm of zero or negative values
    if offset is None:
        offset = 0.
    x_bins = np.logspace(np.log(np.min(np.nanmean(x_array_list, axis=0)) + offset),
                         np.log(np.max(np.nanmean(x_array_list, axis=0)) + offset), bins)
    y_bins = np.logspace(np.log(np.min(np.nanmean(y_array_list, axis=0)) + offset),
                         np.log(np.max(np.nanmean(y_array_list, axis=0)) + offset), bins)

    hist_list = []
    edges_x_list = []
    edges_y_list = []

    for i in range(len(x_array_list)):
        hist, edges_x, edges_y = np.histogram2d(x_array_list[i][~np.isnan(x_array_list[i])],
                                                y_array_list[i][~np.isnan(
                                                    y_array_list[i])],
                                                bins=(x_bins, y_bins))
        hist_list.append(hist)
        edges_x_list.append(edges_x)
        edges_y_list.append(edges_y)

    # Create the 2D histogram
    fig, ax = plt.subplots(dpi=dpi)
    counts = np.mean(hist_list, axis=0)
    xedges = np.mean(edges_x_list, axis=0)
    # FIXME: should this be done after the log?
    yedges = np.mean(edges_y_list, axis=0)

    plt.pcolormesh(xedges, yedges, counts.T, cmap=plt.cm.jet,
                   norm=LogNorm(vmin=1, vmax=np.max(counts)))  # FIXME: here it may fail if the counts are all zeros
    plt.colorbar()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    if title is not None:
        ax.set_title(title)
    if output_path is not None:
        plt.savefig(output_path)
        print(f"2D histogram saved as {output_path}.")
        plt.close()
    else:
        plt.show()
