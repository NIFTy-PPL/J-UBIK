# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from functools import reduce

import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .data import Domain
from .convolve import convolve

try:
    import ducc0.healpix as ducc_hp
except ImportError as err:
    ducc_hp = None
    _DUCC_IMPORT_ERROR = err
else:
    _DUCC_IMPORT_ERROR = None

def plot_result(array,
                domains=None,
                output_file=None,
                logscale=False,
                title=None,
                colorbar=True,
                figsize=(8, 8),
                dpi=100,
                cbar_formatter=None,
                n_rows=None,
                n_cols=None,
                adjust_figsize=False,
                common_colorbar=False,
                share_x=True,
                share_y=True,
                pause_time=None,
                **kwargs):
    """
    Plot a 2D array using imshow() from the matplotlib library.

    Parameters
    ----------
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
        Whether to use the same color bar for all images. Overrides vmin and
        vmax.
    share_x : bool, optional
        Whether to share the x axis.
    share_y : bool, optional
        Whether to share the y axis.
    pause_time : float, optional
        The time in seconds to pause between each plot.
        If None, no pause is performed.
    kwargs : dict, optional
        Additional keyword arguments to pass to imshow().

    Returns
    -------
    None
    """

    shape_len = array.shape
    if len(shape_len) < 2 or len(shape_len) > 3:
        raise ValueError("Wrong input shape for array plot!")
    if len(shape_len) == 2:
        array = array[np.newaxis, :, :]

    n_plots = array.shape[0]

    if n_rows is None:
        n_rows = _get_n_rows_from_n_samples(n_plots)

    if n_cols is None:
        if n_plots % n_rows == 0:
            n_cols = n_plots // n_rows
        else:
            n_cols = n_plots // n_rows + 1

    if adjust_figsize:
        x = int(n_cols / n_rows)
        y = int(n_rows / n_cols)
        if x == 0:
            x = 1
        if y == 0:
            y = 1
        figsize = (x * figsize[0], y * figsize[1])

    n_ax = n_rows * n_cols
    n_del = n_ax - n_plots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize,
                             dpi=dpi, sharex=share_x,
                             sharey=share_y)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    pltargs = {"origin": "lower", "cmap": "viridis"}

    # Handle vmin and vmax
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)

    if colorbar and common_colorbar:
        vmin = min(np.min(array[i]) for i in range(n_plots))
        vmax = max(np.max(array[i]) for i in range(n_plots))

    if logscale:
        if vmin is not None and float(vmin) == 0.:
            vmin = 1e-18  # to prevent LogNorm throwing errors

        pltargs["norm"] = "log"

    kwargs.update({'vmin': vmin, 'vmax': vmax})

    for i in range(n_plots):
        if array[i].ndim != 2:
            raise ValueError("All arrays to plot must be 2-dimensional!")

        if domains is not None:
            half_fov = domains[i]["distances"][0] * domains[i]["shape"][
                0] / 2.0 / 60  # conv to arcmin FIXME: works only for square
            # array
            pltargs["extent"] = [-half_fov, half_fov] * 2
            axes[i].set_xlabel("FOV [arcmin]")
            axes[i].set_ylabel("FOV [arcmin]")

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
        fig.delaxes(axes[n_plots + i])
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_file}.")
        plt.cla()
        plt.clf()
        plt.close()
    else:
        if pause_time is not None:
            plt.pause(pause_time)
            plt.show()
            plt.close()
        else:
            plt.show()


def _infer_healpix_nside(npix: int) -> int:
    """Infer HEALPix NSIDE from the number of pixels."""
    nside = int(round(math.sqrt(npix / 12)))

    if 12 * nside**2 != npix:
        raise ValueError(
            "Input does not look like a HEALPix map. "
            f"Got npix={npix}, but 12*nside**2 does not match."
        )

    return nside


def _ducc_ang2pix(base, theta, phi):
    """Call ducc0 HEALPix ang2pix robustly for different ducc0 versions."""
    theta_flat = np.ravel(theta)
    phi_flat = np.ravel(phi)

    try:
        pix = base.ang2pix(theta_flat, phi_flat)
    except TypeError:
        pix = base.ang2pix(np.column_stack([theta_flat, phi_flat]))

    return np.asarray(pix).reshape(theta.shape)


def _mollweide_imgdata_from_healpix(
    hp_data,
    *,
    xsize: int = 1000,
    flip: str = "astro",
):
    """Project a RING-ordered HEALPix map to a Mollweide image array.

    This follows the PLD-style plotting approach: convert the HEALPix map to a
    regular 2D image and display it with ``imshow``. This gives better control
    over colorbars, logarithmic scaling, clipping, and multi-panel layouts than
    ``healpy.mollview``.

    Parameters
    ----------
    hp_data : array_like
        One HEALPix map with shape ``(12*nside**2,)`` in RING ordering.
    xsize : int
        Horizontal image size in pixels.
    flip : str
        Longitude convention. ``"astro"`` flips longitude so the display follows
        the usual astronomical left-right convention. ``"geo"`` leaves longitude
        unflipped.

    Returns
    -------
    img : np.ndarray
        Projected 2D image array.
    """
    if ducc_hp is None:
        raise ImportError(
            "ducc0 is required for PLD-style HEALPix plotting."
        ) from _DUCC_IMPORT_ERROR

    hp_data = np.asarray(hp_data)
    hp_data = np.ravel(hp_data)

    nside = _infer_healpix_nside(hp_data.size)
    base = ducc_hp.Healpix_Base(nside, "RING")

    ysize = xsize // 2

    xlim = 2.0 * np.sqrt(2.0)
    ylim = np.sqrt(2.0)

    x = np.linspace(-xlim, xlim, xsize)
    y = np.linspace(-ylim, ylim, ysize)
    xx, yy = np.meshgrid(x, y)

    img = np.full((ysize, xsize), np.nan, dtype=float)

    inside = (xx / xlim) ** 2 + (yy / ylim) ** 2 <= 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        aux = np.arcsin(np.clip(yy / np.sqrt(2.0), -1.0, 1.0))
        cos_aux = np.cos(aux)

        lat = np.arcsin(
            np.clip((2.0 * aux + np.sin(2.0 * aux)) / np.pi, -1.0, 1.0)
        )
        lon = np.pi * xx / (2.0 * np.sqrt(2.0) * cos_aux)

    valid = inside & np.isfinite(lat) & np.isfinite(lon)

    if flip == "astro":
        lon = -lon
    elif flip != "geo":
        raise ValueError(f"Unsupported flip={flip!r}. Use 'astro' or 'geo'.")

    theta = 0.5 * np.pi - lat
    phi = np.mod(lon, 2.0 * np.pi)

    pix = _ducc_ang2pix(base, theta[valid], phi[valid])
    img[valid] = hp_data[pix]

    return img


def _as_healpix_stack(data):
    """Normalize input to shape ``(n_maps, npix)``."""
    arr = np.asarray(data)

    if arr.ndim == 1:
        return arr[None, :]

    if arr.ndim == 2:
        return arr

    return arr.reshape((-1, arr.shape[-1]))


def _normalize_titles(title, n_maps):
    if title is None:
        return [None] * n_maps

    if isinstance(title, str):
        if n_maps == 1:
            return [title]
        return [f"{title} [{i}]" for i in range(n_maps)]

    titles = list(title)
    if len(titles) != n_maps:
        raise ValueError(
            f"Expected {n_maps} titles, got {len(titles)}."
        )

    return titles


def _finite_values_for_limits(values, *, logscale: bool):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if logscale:
        values = values[values > 0.0]

    return values


def _get_limits(values, *, logscale: bool, vmin=None, vmax=None, percentile=None):
    finite = _finite_values_for_limits(values, logscale=logscale)

    if finite.size == 0:
        if logscale:
            return 1.0e-30, 1.0
        return 0.0, 1.0

    if percentile is not None:
        pmin, pmax = percentile
        auto_vmin, auto_vmax = np.nanpercentile(finite, [pmin, pmax])
    else:
        auto_vmin = np.nanmin(finite)
        auto_vmax = np.nanmax(finite)

    if vmin is None:
        vmin = auto_vmin
    if vmax is None:
        vmax = auto_vmax

    if logscale:
        vmin = max(float(vmin), np.nanmin(finite[finite > 0.0]))
        vmax = max(float(vmax), vmin * (1.0 + 1.0e-12))

    if not logscale and vmax <= vmin:
        vmax = vmin + 1.0

    return float(vmin), float(vmax)


def _plot_projected_healpix_panel(
    fig,
    ax,
    hp_map,
    *,
    title=None,
    unit=None,
    logscale=False,
    vmin=None,
    vmax=None,
    percentile=None,
    cmap="viridis",
    xsize=1000,
    flip="astro",
    show_colorbar=True,
):
    img = _mollweide_imgdata_from_healpix(
        hp_map,
        xsize=xsize,
        flip=flip,
    )

    if logscale:
        img = img.copy()
        img[img <= 0.0] = np.nan
        vmin, vmax = _get_limits(
            img,
            logscale=True,
            vmin=vmin,
            vmax=vmax,
            percentile=percentile,
        )
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = _get_limits(
            img,
            logscale=False,
            vmin=vmin,
            vmax=vmax,
            percentile=percentile,
        )
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        img,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            fraction=0.046,
            pad=0.04,
        )
        if unit is not None:
            cbar.set_label(unit)

        if logscale:
            cbar.formatter = LogFormatterMathtext()
            cbar.update_ticks()

    return im


# -------------------------------------------------------------------
# PLD-style HEALPix plotting
# -------------------------------------------------------------------

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext

try:
    import ducc0.healpix as ducc_hp
except ImportError as err:
    ducc_hp = None
    _DUCC_IMPORT_ERROR = err
else:
    _DUCC_IMPORT_ERROR = None


def _infer_healpix_nside(npix: int) -> int:
    """Infer HEALPix NSIDE from a RING HEALPix map length."""
    nside = int(round(math.sqrt(npix / 12)))

    if 12 * nside**2 != npix:
        raise ValueError(
            "Input does not look like a HEALPix map. "
            f"Got npix={npix}, but no integer nside satisfies "
            "npix = 12*nside**2."
        )

    return nside


def _ducc_ang2pix(base, theta, phi):
    """Call ducc0 HEALPix ang2pix robustly across ducc0 versions."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    theta_flat = theta.ravel()
    phi_flat = phi.ravel()

    try:
        pix = base.ang2pix(theta_flat, phi_flat)
    except TypeError:
        pix = base.ang2pix(np.column_stack([theta_flat, phi_flat]))

    return np.asarray(pix).reshape(theta.shape)


def _mollweide_imgdata_from_healpix(
    hp_data,
    *,
    xsize: int = 1000,
    flip: str = "astro",
):
    """Project one RING HEALPix map to a regular Mollweide image.

    This follows the PLD/reconstruction plotting style: first sample the
    HEALPix map on a regular projected image grid, then display the result with
    ``imshow``. This gives stable multi-panel layouts and full matplotlib
    control over logarithmic colorbars.
    """
    if ducc_hp is None:
        raise ImportError(
            "ducc0 is required for PLD-style HEALPix plotting."
        ) from _DUCC_IMPORT_ERROR

    hp_data = np.asarray(hp_data, dtype=float).ravel()

    nside = _infer_healpix_nside(hp_data.size)
    base = ducc_hp.Healpix_Base(nside, "RING")

    ysize = xsize // 2

    xlim = 2.0 * np.sqrt(2.0)
    ylim = np.sqrt(2.0)

    x = np.linspace(-xlim, xlim, xsize)
    y = np.linspace(-ylim, ylim, ysize)
    xx, yy = np.meshgrid(x, y)

    img = np.full((ysize, xsize), np.nan, dtype=float)

    inside = (xx / xlim) ** 2 + (yy / ylim) ** 2 <= 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        aux = np.arcsin(np.clip(yy / np.sqrt(2.0), -1.0, 1.0))
        cos_aux = np.cos(aux)

        lat = np.arcsin(
            np.clip(
                (2.0 * aux + np.sin(2.0 * aux)) / np.pi,
                -1.0,
                1.0,
            )
        )

        lon = np.pi * xx / (2.0 * np.sqrt(2.0) * cos_aux)

    valid = inside & np.isfinite(lat) & np.isfinite(lon)

    if flip == "astro":
        lon = -lon
    elif flip != "geo":
        raise ValueError(f"Unsupported flip={flip!r}. Use 'astro' or 'geo'.")

    theta = 0.5 * np.pi - lat
    phi = np.mod(lon, 2.0 * np.pi)

    pix = _ducc_ang2pix(base, theta[valid], phi[valid])
    img[valid] = hp_data[pix]

    return img


def _as_healpix_stack(data):
    """Normalize HEALPix input to shape ``(n_maps, npix)``."""
    arr = np.asarray(data)

    if arr.ndim == 1:
        return arr[None, :]

    if arr.ndim == 2:
        return arr

    return arr.reshape((-1, arr.shape[-1]))


def _normalize_titles(title, n_maps):
    if title is None:
        return [None] * n_maps

    if isinstance(title, str):
        if n_maps == 1:
            return [title]
        return [f"{title} [{i}]" for i in range(n_maps)]

    titles = list(title)

    if len(titles) != n_maps:
        raise ValueError(f"Expected {n_maps} titles, got {len(titles)}.")

    return titles


def _valid_values_for_limits(values, *, logscale: bool):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if logscale:
        values = values[values > 0.0]

    return values


def _get_color_limits(
    values,
    *,
    logscale: bool,
    vmin=None,
    vmax=None,
    percentile=None,
):
    finite = _valid_values_for_limits(values, logscale=logscale)

    if finite.size == 0:
        if logscale:
            return 1.0e-30, 1.0
        return 0.0, 1.0

    if percentile is not None:
        auto_vmin, auto_vmax = np.nanpercentile(finite, percentile)
    else:
        auto_vmin = np.nanmin(finite)
        auto_vmax = np.nanmax(finite)

    if vmin is None:
        vmin = auto_vmin

    if vmax is None:
        vmax = auto_vmax

    vmin = float(vmin)
    vmax = float(vmax)

    if logscale:
        min_positive = float(np.nanmin(finite[finite > 0.0]))
        vmin = max(vmin, min_positive)

        if vmax <= vmin:
            vmax = vmin * (1.0 + 1.0e-12)

    else:
        if vmax <= vmin:
            vmax = vmin + 1.0

    return vmin, vmax


def _style_colorbar(cbar, *, dark_background: bool):
    if not dark_background:
        return

    cbar.ax.xaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("white")


def _plot_single_healpix_panel(
    fig,
    ax,
    cax,
    hp_map,
    *,
    title=None,
    unit=None,
    logscale=False,
    vmin=None,
    vmax=None,
    percentile=None,
    cmap="inferno",
    xsize=1000,
    flip="astro",
    dark_background=True,
):
    """Plot one HEALPix panel with the exact PLD-style colorbar layout."""
    img = _mollweide_imgdata_from_healpix(
        hp_map,
        xsize=xsize,
        flip=flip,
    )

    cmap_obj = plt.get_cmap(cmap).copy()

    if dark_background:
        cmap_obj.set_bad("black")
        ax.set_facecolor("black")
        cax.set_facecolor("black")
    else:
        cmap_obj.set_bad("white")

    if logscale:
        img = img.copy()
        img[img <= 0.0] = np.nan

        vmin, vmax = _get_color_limits(
            img,
            logscale=True,
            vmin=vmin,
            vmax=vmax,
            percentile=percentile,
        )

        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = _get_color_limits(
            img,
            logscale=False,
            vmin=vmin,
            vmax=vmax,
            percentile=percentile,
        )

        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        img,
        origin="lower",
        interpolation="nearest",
        cmap=cmap_obj,
        norm=norm,
    )

    ax.set_anchor("S")
    ax.set_axis_off()
    
    if title is not None:
        title_kwargs = dict(
            fontsize=9,
            pad=4,
        )

        if dark_background:
            title_kwargs["color"] = "white"

    ax.set_title(title, **title_kwargs)

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )

    if unit is not None:
        cbar.set_label(unit)

    if logscale:
        cbar.formatter = LogFormatterMathtext()
        cbar.update_ticks()

    _style_colorbar(cbar, dark_background=dark_background)

    return im, cbar


def plot_healpix_result(
    data,
    *,
    n_rows=1,
    n_cols=1,
    figsize=None,
    title=None,
    unit=None,
    logscale=False,
    common_colorbar=False,
    vmin=None,
    vmax=None,
    percentile=None,
    cmap="viridis",
    xsize=1000,
    flip="astro",
    dark_background=True,
):
    """Plot HEALPix maps using the PLD/reconstruction plotting style.

    This routine intentionally follows the style used in the Fermi-LAT PLD
    response-inspection plots: a projected HEALPix image is shown with
    ``imshow`` and every panel has a dedicated horizontal colorbar row directly
    below it. This avoids the ``healpy.mollview`` layout and gives stable,
    publication/debugging-style multi-panel figures.

    Parameters
    ----------
    data : array_like
        HEALPix map or stack of maps. Accepted shapes are ``(npix,)``,
        ``(n_maps, npix)``, or any higher-dimensional array whose last axis is
        the HEALPix pixel axis.
    n_rows, n_cols : int
        Map panel layout.
    figsize : tuple, optional
        Matplotlib figure size.
    title : str or sequence of str, optional
        Panel title or list of panel titles.
    unit : str, optional
        Colorbar label.
    logscale : bool
        If true, use logarithmic color normalization.
    common_colorbar : bool
        If true, use common color limits across panels. The layout still keeps
        one colorbar per panel, matching the PLD-style response plots.
    vmin, vmax : float, optional
        Explicit color limits.
    percentile : tuple[float, float], optional
        Percentile limits for automatic color scaling, e.g. ``(1, 99.9)``.
        Useful for heavy-tailed inverse-gamma point-source samples.
    cmap : str
        Matplotlib colormap. Default is ``"inferno"``, matching the response
        diagnostic style.
    xsize : int
        Horizontal resolution of the projected Mollweide image.
    flip : str
        Longitude convention, either ``"astro"`` or ``"geo"``.
    dark_background : bool
        If true, use the black-background PLD diagnostic style.

    Returns
    -------
    fig, axes
        Matplotlib figure and map axes array.
    """
    maps = _as_healpix_stack(data)
    n_maps = maps.shape[0]

    if n_rows * n_cols < n_maps:
        raise ValueError(
            f"Layout has {n_rows * n_cols} panels, but data has {n_maps} maps."
        )

    titles = _normalize_titles(title, n_maps)

    if figsize is None:
        figsize = (4.8 * n_cols, 3.2 * n_rows)

    # Every map row gets a colorbar row below it. This is the same visual logic
    # as the response-inspection plots.
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([1.0, 0.055])

    fig = plt.figure(figsize=figsize)

    if dark_background:
        fig.patch.set_facecolor("black")

    gs = fig.add_gridspec(
        2 * n_rows,
        n_cols,
        height_ratios=height_ratios,
        hspace=0.04,
        wspace=0.08,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    cbar_axes = np.empty((n_rows, n_cols), dtype=object)

    shared_vmin = vmin
    shared_vmax = vmax

    if common_colorbar:
        shared_vmin, shared_vmax = _get_color_limits(
            maps,
            logscale=logscale,
            vmin=vmin,
            vmax=vmax,
            percentile=percentile,
        )

    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols

        ax = fig.add_subplot(gs[2 * row, col])
        cax = fig.add_subplot(gs[2 * row + 1, col])
        ax.set_anchor("S")
        cax.set_anchor("N")

        axes[row, col] = ax
        cbar_axes[row, col] = cax

        if i >= n_maps:
            ax.set_axis_off()
            cax.set_axis_off()
            continue

        panel_vmin = shared_vmin if common_colorbar else vmin
        panel_vmax = shared_vmax if common_colorbar else vmax

        _plot_single_healpix_panel(
            fig,
            ax,
            cax,
            maps[i],
            title=titles[i],
            unit=unit,
            logscale=logscale,
            vmin=panel_vmin,
            vmax=panel_vmax,
            percentile=None if common_colorbar else percentile,
            cmap=cmap,
            xsize=xsize,
            flip=flip,
            dark_background=dark_background,
        )

    return fig, axes


def plot_histograms(hist,
                    edges,
                    filename,
                    logx=False,
                    logy=False,
                    title=None):
    """
    Plots a histogram and saves it to a file.

    This function creates a bar plot from histogram data and optionally applies
    logarithmic scaling to the x-axis and/or y-axis.
    The plot is saved to the specified file.

    Parameters
    ----------
    hist : array-like
        The values of the histogram bars.
        Should be a one-dimensional array-like object.
    edges : array-like
        The bin edges of the histogram.
        Should be a one-dimensional array-like object with length
        one more than `hist`.
    filename : str
        The path where the histogram plot will be saved.
        Should include the file extension (e.g., '.png', '.pdf').
    logx : bool, optional
        If True, the x-axis will be scaled logarithmically.
        Default is False.
    logy : bool, optional
        If True, the y-axis will be scaled logarithmically.
        Default is False.
    title : str, optional
        The title of the histogram plot.
        If None, no title will be displayed.
        Default is None.

    Returns
    -------
    None
        The function does not return any value. It saves the plot to the
        specified file.

    Notes
    -----
    - Ensure that `hist` and `edges` are compatible with each other. `edges`
    should have
      one more element than `hist`.
    - The file specified by `filename` will be overwritten if it already exists.
    """
    plt.bar(edges[:-1], hist, width=edges[1] - edges[0], align='edge')

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    if title:
        plt.title(title)

    plt.savefig(filename)
    plt.cla()
    plt.clf()
    plt.close()
    print(f"Histogram saved as {filename}.")


def plot_sample_averaged_log_2d_histogram(x_array_list,
                                          x_label,
                                          y_array_list,
                                          y_label,
                                          x_lim=None,
                                          y_lim=None,
                                          bins=100,
                                          dpi=400,
                                          title=None,
                                          output_path=None,
                                          offset=0.,
                                          figsize=None):
    """ Plot a 2d histogram for the arrays given for x_array and y_array.


    Parameters
    ----------
    x_array_list : list of numpy.ndarray
        list of samples of x_axis array of 2d-histogram
    x_label : string
        x-axis label of the 2d-histogram
    y_array_list : numpy.ndarray
        list of samples of y-axis array of 2d-histogram
    y_label : string
        y-axis label of the 2d-histogram
    x_lim : tuple, optional
        Limits of the x-axis
    y_lim : tuple, optional
        Limits of the y-axis
    bins : int
        Number of bins of the 2D-histogram
    dpi : int, optional
        Resolution of the figure
    title : string, optional
        Title of the 2D histogram
    output_path : string, optional
        Output directory for the plot.
        If None (Default) the plot is not saved.
    offset : float, optional
        Offset for the logarithmic binning.
        Default is 0.
    figsize : tuple, optional
        Size of the figure

    Returns
    -------
    None
    """
    if len(x_array_list) != len(y_array_list):
        raise ValueError('Need same number of samples for x- and y-axis.')

    x_bins = np.logspace(
        np.log(np.min(np.nanmean(x_array_list, axis=0)) + offset),
        np.log(np.max(np.nanmean(x_array_list, axis=0)) + offset), bins)
    y_bins = np.logspace(
        np.log(np.min(np.nanmean(y_array_list, axis=0)) + offset),
        np.log(np.max(np.nanmean(y_array_list, axis=0)) + offset), bins)

    hist_list = []
    edges_x_list = []
    edges_y_list = []

    for i in range(len(x_array_list)):
        hist, edges_x, edges_y = np.histogram2d(
            x_array_list[i][~np.isnan(x_array_list[i])],
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
    plt.pcolormesh(xedges, yedges, counts.T, cmap=plt.cm.jet,
                   norm=LogNorm(vmin=1, vmax=np.max(counts)))

    plt.figure(figsize=figsize)
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
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_rgb(array,
             sat_min=[0, 0, 0],
             sat_max=[1, 1, 1],
             name=None,
             sigma=None,
             log=False,
             pause_time=None,
             ):
    """
    Plots an RGB image and saves it to a file.

    This function processes an RGB image array, applies optional smoothing,
    clipping, and logarithmic scaling, and then saves the image to a PNG file.

    Parameters
    ----------
    array : ndarray
        An array with shape (RGB, Space, Space) representing the RGB image data.
        The first dimension should correspond to the color channels
        (Red, Green, Blue).
    sat_min : list of float, optional
        Minimum values for saturation clipping in each color channel.
        Should be a list with three elements corresponding to the RGB channels.
        Default is [0, 0, 0].
    sat_max : list of float, optional
        Maximum values for saturation clipping in each color channel.
        Should be a list with three elements corresponding to the RGB channels.
        Default is [1, 1, 1].
    name : str, optional
        The base name of the file where the plot will be saved.
        The file extension '.png' will be added automatically.
        If None, no file will be saved.
    sigma : float or None, optional
        Standard deviation for Gaussian smoothing.
        If None, no smoothing is applied. Default is None.
    log : bool, optional
        If True, apply logarithmic scaling to the
        image data (non-zero values only). Default is False.
    pause_time : float, optional
        The time in seconds to pause between each plot.
        If None, no pause is applied. Default is None.

    Returns
    -------
    None
        The function saves the RGB image to a PNG file and does not
        return any value.

    Notes
    -----
    - The image will be saved with the filename format '<name>.png'.
    - Ensure that the input array is correctly formatted with the first
    dimension as RGB channels.
    """
    if sigma is not None:
        array = _smooth(sigma, array)
    if sat_min is not None and sat_max is not None:
        array = _clip(array, sat_min, sat_max)
    if log:
        array = _non_zero_log(array)

    array = np.moveaxis(array, 0, -1)  # Move the RGB dimension
    # to the last axis for plotting
    plot_data = _norm_rgb_plot(array)  # Normalize data for RGB plotting
    plt.imshow(plot_data, origin="lower")

    if name is not None:
        plt.savefig(name + ".png", dpi=500) # TODO: make dpi configurable
        plt.cla()
        plt.clf()
        plt.close()
        print(f"RGB image saved as {name}.png")
    else:
        if pause_time is not None:
            plt.pause(pause_time)
            plt.show()
            plt.close()
        else:
            plt.show()


def _get_n_rows_from_n_samples(n_samples):
    """
    A function to get the number of rows from the given number of samples.

    Parameters
    ----------
    n_samples: int
        number of samples

    Returns
    -------
    number of rows: int
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


def _norm_rgb_plot(x):
    plot_data = np.zeros(x.shape)
    x = np.array(x)
    # norm on RGB to 0-1
    for i in range(3):
        a = x[:, :, i]
        if a[a != 0].size == 0:
            minim = 0
            maxim = 0
        else:
            minim = a[a != 0].min()
            maxim = a[a != 0].max()
        a[a != 0] = (a[a != 0] - minim) / (maxim - minim)
        plot_data[:, :, i] = a
    return plot_data


def _gauss(x, y, sig):
    """Evaluate 2D Normal distribution."""
    const = 1 / (np.sqrt(2 * np.pi * sig ** 2))
    r = np.sqrt(x ** 2 + y ** 2)
    f = const * np.exp(-r ** 2 / (2 * sig ** 2))
    return f


def get_gaussian_kernel(domain, sigma):
    """Get 2D Gaussian kernel for fft convolution.

    Parameters
    ----------
    domain: jubik.Domain
        domain of the Gaussian kernel
    sigma: float
        standard deviation of the Gaussian

    Returns
    -------
    Gauss kernel: 2D array
    """
    border = (domain.shape * domain.distances // 2)
    x = np.linspace(-border[0], border[0], domain.shape[0])
    y = np.linspace(-border[1], border[1], domain.shape[1])
    xv, yv = np.meshgrid(x, y)
    kern = _gauss(xv, yv, sigma)
    kern = np.fft.fftshift(kern)
    dvol = reduce(lambda a, b: a * b, domain.distances)
    normalization = kern.sum() * dvol
    kern = kern * normalization ** -1
    return kern.T


def _smooth(sig, x):
    domain = Domain(x.shape, np.ones([3]))
    gauss_domain = Domain(x.shape[1:], np.ones([2]))

    smoothing_kernel = get_gaussian_kernel(gauss_domain, sig)
    smoothing_kernel = smoothing_kernel[np.newaxis, ...]
    smooth_data = convolve(x, smoothing_kernel, domain, [1, 2])
    return np.array(smooth_data)


def _clip(x, sat_min, sat_max):
    clipped = np.zeros(x.shape)
    print("Change the Saturation")
    for i in range(3):
        clipped[i] = np.clip(x[i], a_min=sat_min[i], a_max=sat_max[i])
        clipped[i] = clipped[i] - sat_min[i]
    return clipped


def _non_zero_log(x):
    x_arr = np.array(x)
    log_x = np.zeros(x_arr.shape)
    log_x[x_arr > 0] = np.log(x_arr[x_arr > 0])
    return log_x
