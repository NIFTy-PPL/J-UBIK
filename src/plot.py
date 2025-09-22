# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from functools import reduce
from typing import Optional, Sequence, List, Tuple, Union
from dataclasses import dataclass
import math

from jax import vmap
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from .data import Domain
from .convolve import convolve
from nifty.re import logger



def display_plot_or_save(
        fig: plt.Figure, 
        filename: Optional[str], 
        *,
        dpi: int, 
        bbox_inches=None, 
):
    """Save *this* figure if filename is given, else show it. Optionally log and close."""
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        plt.cla()
        plt.clf()
        logger.info(f"Plot saved to {filename}.")
    else:
        plt.show()


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
            # divider = make_axes_locatable(axes[i])
            # cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, ax=axes[i], format=cbar_formatter)
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

    Returns:
    --------
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


@dataclass
class RGBScaleConfig:
    """Encapsulate RGB normalization strategy and optional relative channel weights."""

    mode: str
    relative_scale: Optional[np.ndarray] = None

    def __post_init__(self):
        self.mode = str(self.mode).lower()
        if self.mode not in {"global", "per_channel"}:
            raise ValueError("scale_mode must be 'global' or 'per_channel'.")

        if self.relative_scale is not None:
            rel = np.asarray(self.relative_scale, dtype=float)
            if rel.shape != (3,):
                raise ValueError("relative_scale must be a 3-element sequence.")
            if np.any(~np.isfinite(rel)):
                raise ValueError("relative_scale entries must be finite numbers.")
            if np.any(rel < 0):
                raise ValueError("relative_scale entries must be non-negative.")
            if np.all(rel <= 0):
                raise ValueError("relative_scale must contain at least one positive value.")
            self.relative_scale = rel

    @classmethod
    def from_settings(cls, value: Union[str, Sequence[float], 'RGBScaleConfig']) -> 'RGBScaleConfig':
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(mode=value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return cls(mode="per_channel", relative_scale=np.asarray(value, dtype=float))
        raise TypeError("scale_mode must be a string, a 3-element sequence, or RGBScaleConfig.")

    def normalize(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """Normalize clipped RGB values according to the configured mode."""

        def _nanmax_safe(arr: np.ndarray) -> float:
            try:
                return float(np.nanmax(arr))
            except ValueError:
                return float("nan")

        if self.mode == "global":
            global_max = _nanmax_safe(rgb) if rgb.size else float("nan")
            if (not np.isfinite(global_max)) or global_max <= 0:
                img = np.zeros_like(rgb, dtype=float)
            else:
                img = rgb / global_max
            return img, {
                "scale_mode": "global",
                "global_max": global_max,
                "per_channel_max": None,
                "relative_scale": None,
                "relative_scale_normalized": None,
            }

        # per-channel scaling
        img = np.empty_like(rgb, dtype=float)
        channel_max = np.zeros(3, dtype=float)
        for c in range(3):
            ch = rgb[c]
            ch_max = _nanmax_safe(ch) if ch.size else float("nan")
            channel_max[c] = ch_max if np.isfinite(ch_max) else float("nan")
            if not np.isfinite(ch_max) or ch_max <= 0:
                img[c] = np.zeros_like(ch, dtype=float)
            else:
                img[c] = ch / ch_max

        info = {
            "scale_mode": "per_channel",
            "per_channel_max": channel_max.tolist(),
            "global_max": None,
            "relative_scale": None,
            "relative_scale_normalized": None,
        }

        if self.relative_scale is not None:
            weights = self.relative_scale
            max_weight = float(np.max(weights))
            if not np.isfinite(max_weight) or max_weight <= 0:
                raise ValueError("relative_scale must contain at least one positive finite entry.")
            normalized_weights = weights / max_weight
            for c in range(3):
                img[c] *= normalized_weights[c]
            info["relative_scale"] = weights.tolist()
            info["relative_scale_normalized"] = normalized_weights.tolist()

        return img, info


def plot_rgb(array,
             sat_min=[0, 0, 0],
             sat_max=[1, 1, 1],
             sigma=None,
             log=False,
             *,
             # new: pass-through args for spectral→RGB conversion
             rgb_energies_existing=None,
             rgb_energies_target=None,
             rgb_log_spacing: bool = True,
             rgb_method: str = "linear",   # "linear" | "cubic"
             # plotting controls
             scale_mode: str | Sequence[float] | RGBScaleConfig = "global",
             show_flux_bars: bool = False,
             flux_bar_decimals: int = 3,
             scalebar_px: int | None = None,
             px_scale: float | None = None,
             scalebar_label: str | None = None,
             scalebar_loc: str = "lower right",
             ax: plt.Axes | None = None,
             name=None,
             dpi=300,
             bbox_inches=None,
             verbose: bool = True,            # new: log when converting to RGB
             ):
    """
    Plot an RGB image with optional spectral conversion, clipping, and annotations.

    The function accepts RGB images or spectral cubes and converts cubes to RGB via
    `to_rgb_bands` before applying per-channel flux clipping, optional smoothing/log
    scaling, and display overlays such as flux scales or scalebars.

    Parameters
    ----------
    array : np.ndarray
        Input image data with shape (3, M, Q), (M, Q, 3), or (N, M, Q) for spectral cubes.
    sat_min : float or Sequence[float], optional
        Lower cumulative-flux quantile(s) per channel used before normalization (value(s) in [0, 1]).
    sat_max : float or Sequence[float], optional
        Upper cumulative-flux quantile(s) per channel used before normalization (value(s) in [0, 1]).
    sigma : float or None, optional
        Standard deviation of the Gaussian smoothing applied after RGB conversion.
    log : bool, optional
        Apply a natural logarithm to positive pixels after smoothing.
    rgb_energies_existing : array-like or None, optional
        Energies/frequencies associated with the input spectral channels when converting.
    rgb_energies_target : array-like of length 3 or None, optional
        Target energies for the RGB bands when performing spectral conversion.
    rgb_log_spacing : bool, optional
        Assume log-spaced channels when `rgb_energies_existing` is not provided.
    rgb_method : {"linear", "cubic"}, optional
        Interpolation method used by `to_rgb_bands` during spectral conversion.
    scale_mode : {"global", "per_channel"} or Sequence[float], optional
        Normalization strategy. A 3-sequence triggers per-channel scaling with relative
        weights applied after normalization (see `relative_scale_normalized` in the return info).
    show_flux_bars : bool, optional
        Draw inset color bars that visualize the clipping thresholds per channel.
    flux_bar_decimals : int, optional
        Number of decimals shown on the flux-bar tick labels.
    scalebar_px : int or None, optional
        Width of the scalebar in pixels; omitted when None.
    px_scale : float or None, optional
        Physical scale per pixel used to annotate the scalebar label.
    scalebar_label : str or None, optional
        Custom text for the scalebar; defaults to a generated label when omitted.
    scalebar_loc : str, optional
        Location code passed to `AnchoredSizeBar` for the scalebar.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on; a new figure/axes is created when None.
    name : str or None, optional
        Output path used to save the figure when a new figure is created.
    dpi : int, optional
        Resolution in dots per inch used when saving a newly created figure.
    bbox_inches : str or None, optional
        Bounding box passed to `display_plot_or_save` while saving.
    verbose : bool, optional
        Log diagnostic messages during axis reordering or spectral conversion.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure that contains the rendered RGB image.
    ax : matplotlib.axes.Axes
        Axes used for plotting the RGB image.
    info : dict
        Metadata describing the applied scaling, including `rgb_energies` when available.

    Raises
    ------
    ValueError
        If `array` is not three-dimensional or if `scale_mode` is unsupported.
    """

    # ---------------- helpers ----------------
    def _as_triplet(x):
        if isinstance(x, (int, float)):
            return [x, x, x]
        if len(x) != 3:
            raise ValueError("sat_min/sat_max must be a float or a list of 3 floats (per RGB channel).")
        return x

    def _flux_quantile_threshold(ch: np.ndarray, q: float) -> float:
        x = np.asarray(ch, dtype=float).ravel()
        if x.size == 0:
            return 0.0
        shift = x.min()
        x_shift = x - shift if shift < 0 else x
        tot = x_shift.sum()
        if not np.isfinite(tot) or tot <= 0:
            return float(np.nanmin(ch) if q <= 0 else np.nanmax(ch))
        order = np.argsort(x)
        vals = x[order]
        weights = x_shift[order]
        cflux = np.cumsum(weights) / tot
        idx = np.searchsorted(cflux, np.clip(q, 0.0, 1.0), side="left")
        idx = min(idx, vals.size - 1)
        return float(vals[idx])

    def _per_channel_flux_clip(rgb: np.ndarray, qmin, qmax) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = np.empty_like(rgb, dtype=float)
        vmins = np.zeros(3, dtype=float)
        vmaxs = np.zeros(3, dtype=float)
        for c in range(3):
            ch = rgb[c]
            vmin = _flux_quantile_threshold(ch, qmin[c])
            vmax = _flux_quantile_threshold(ch, qmax[c])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.nanmax(ch))
                vmin = float(np.nanmin(ch))
                if not np.isfinite(vmax) or vmax <= vmin:
                    out[c] = np.zeros_like(ch, dtype=float)
                    vmins[c] = vmin
                    vmaxs[c] = vmax
                    continue
            out[c] = np.clip(ch, vmin, vmax)
            vmins[c], vmaxs[c] = vmin, vmax
        return out, vmins, vmaxs

    def _add_scalebar(ax, length_px: int, px_scale: float | None, label: str | None,
                      loc: str = "lower right", color="white"):
        if label is None:
            if px_scale is not None and np.isfinite(px_scale):
                phys = length_px * px_scale
                label = f"{length_px}px ({phys:g})"
            else:
                label = f"{length_px}px"
        bar = AnchoredSizeBar(ax.transData,
                              length_px, label,
                              loc=loc,
                              pad=0.4,
                              color=color,
                              frameon=True,
                              size_vertical=max(1, int(0.01 * length_px)),
                              fontproperties=fm.FontProperties(size=8))
        if bar.txt_label is not None:
            bar.txt_label.set_color(color)
        if bar.patch is not None:
            bar.patch.set_alpha(0.5)
        ax.add_artist(bar)

    def _add_flux_bars(ax, vmins, vmaxs, decimals=3):
        pad = 0.02
        h = 0.08
        w = 0.5
        left = 0.5 - w / 2
        bottom0 = pad
        cmaps = ["Reds", "Greens", "Blues"]
        for i, (cmap, vmin, vmax) in enumerate(zip(cmaps, vmins, vmaxs)):
            ax_in = ax.inset_axes([left, bottom0 + i*(h+0.01), w, h])
            grad = np.linspace(0, 1, 256)[None, :]
            ax_in.imshow(grad, aspect="auto", cmap=cmap, origin="lower",
                         extent=[0, 1, 0, 1])
            ax_in.set_xticks([0, 1], [f"{vmin:.{decimals}f}", f"{vmax:.{decimals}f}"])
            ax_in.set_yticks([])
            for spine in ax_in.spines.values():
                spine.set_visible(False)
            ax_in.tick_params(axis='x', labelsize=7)
        ax.text(left - 0.02, bottom0 + 3*(h+0.01) - 0.015, "Flux clip\n(vmin→vmax)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color="w",
                bbox=dict(boxstyle="round,pad=0.2", fc=(0,0,0,0.4), ec="none"))

    # --------- possibly convert to RGB first ---------
    arr = np.asarray(array)
    rgb_energies_used = None

    if arr.ndim != 3:
        raise ValueError(f"`array` must be 3D (C/M/N axes); got shape {arr.shape}.")

    # Cases: (3, M, Q), (M, Q, 3), or (N, M, Q) with N != 3
    if arr.shape[0] == 3:
        rgb = arr
    elif arr.shape[-1] == 3:
        # move channels to axis 0
        if verbose:
            try:
                jft.logger.info("Input is (M,Q,3); moving channel axis to front → (3,M,Q).")
            except Exception:
                pass
        rgb = np.moveaxis(arr, -1, 0)
    else:
        # Need conversion from spectral cube → RGB
        if verbose:
            logger.info(f"Input appears to be spectral cube {arr.shape}; converting to RGB via to_rgb_bands.")

        rgb, rgb_energies_used = to_rgb_bands(
            arr,
            energies_existing=rgb_energies_existing,
            energies_target=rgb_energies_target,
            log_spacing=rgb_log_spacing,
            method=rgb_method,
        )

    # --------- optional smoothing / log AFTER RGB conversion ---------
    if sigma is not None:
        rgb = _smooth(sigma, rgb)
    if log:
        rgb = _non_zero_log(rgb)

    # 1) Per-channel clip by cumulative-flux thresholds
    sat_min = [float(np.clip(x, 0.0, 1.0)) for x in _as_triplet(sat_min)]
    sat_max = [float(np.clip(x, 0.0, 1.0)) for x in _as_triplet(sat_max)]
    arr_clipped, vmins, vmaxs = _per_channel_flux_clip(rgb, sat_min, sat_max)

    # 2) Normalize for display
    scale_config = RGBScaleConfig.from_settings(scale_mode)
    img01, scale_details = scale_config.normalize(arr_clipped)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(np.moveaxis(img01, 0, -1), origin="lower")
    ax.set_xticks([]); ax.set_yticks([])

    if scalebar_px is not None:
        _add_scalebar(ax, scalebar_px, px_scale, scalebar_label, scalebar_loc)
    if show_flux_bars:
        _add_flux_bars(ax, vmins, vmaxs, decimals=flux_bar_decimals)

    if created_fig:
        display_plot_or_save(fig, filename=name, dpi=dpi, bbox_inches=bbox_inches)

    info = dict(
        vmins=vmins,
        vmaxs=vmaxs,
        rgb_energies=rgb_energies_used,
        **scale_details,
    )
    return fig, ax, info


def plot_rgb_grid(images: np.ndarray,
                  nrows: int | None = None,
                  ncols: int | None = None,
                  figsize: Tuple[float, float] | None = None,
                  name: str | None = None,
                  dpi: int | None = None,
                  bbox_inches: str | None = None,
                  *,
                  titles: Sequence[str] | None = None,
                  suptitle: str | None = None,
                  wspace: float = 0.05,
                  hspace: float = 0.05,
                  share_axes: bool = True,
                  channel_axis: int = 1,
                  # anything below is passed through to plot_rgb
                  sat_min=[0, 0, 0],
                  sat_max=[1, 1, 1],
                  scale_mode: str = "global",
                  sigma=None,
                  log: bool = False,
                  show_flux_bars: bool = False,
                  scalebar_px: int | None = None,
                  px_scale: float | None = None,
                  scalebar_label: str | None = None,
                  scalebar_loc: str = "lower right",
                  flux_bar_decimals: int = 3,
                  rgb_energies_existing=None,
                  rgb_energies_target=None,
                  rgb_log_spacing: bool = True,
                  rgb_method: str = "linear",
                  verbose: bool = True,
                  ) -> tuple[plt.Figure, np.ndarray, List[dict]]:
    """
    Render a grid of RGB images or spectral cubes by delegating each cell to `plot_rgb`.

    Parameters
    ----------
    images : np.ndarray
        Stack of input images with shape (N, C, H, W) or (N, H, W, C); the channel axis is
        chosen via `channel_axis` and must contain at least three bands (C ≥ 3).
    nrows, ncols : int or None
        Grid layout. If omitted, a near-square arrangement is chosen automatically.
    figsize : tuple[float, float] or None
        Figure size in inches. Defaults to `(ncols*3, nrows*3)` when None.
    name : str or None
        Output path to save the composed grid via `display_plot_or_save`.
    dpi : int or None
        Resolution used when saving the figure (only applied if `name` is given).
    bbox_inches : str or None
        Bounding-box option forwarded to `display_plot_or_save` during saving.
    titles : Sequence[str] or None
        Optional per-panel titles.
    suptitle : str or None
        Figure-wide title drawn above the grid.
    wspace, hspace : float
        Horizontal/vertical spacing passed to `plt.subplots_adjust`.
    share_axes : bool, optional
        Remove ticks on individual panels when True.
    channel_axis : int, optional
        Axis index of the spectral/RGB dimension in `images` (excludes the batch axis).
    sat_min, sat_max : float or Sequence[float], optional
        Saturation quantiles forwarded to `plot_rgb`.
    scale_mode : {"global", "per_channel"}, optional
        Normalization mode passed to `plot_rgb`.
    sigma : float or None, optional
        Gaussian smoothing applied within `plot_rgb`.
    log : bool, optional
        Apply a logarithmic stretch after smoothing inside `plot_rgb`.
    show_flux_bars : bool, optional
        Draw per-channel flux bars inside each panel when True.
    scalebar_px : int or None, optional
        Width of the scalebar annotation; omitted when None.
    px_scale : float or None, optional
        Physical pixel scale for the scalebar label.
    scalebar_label : str or None, optional
        Custom scalebar text passed through to `plot_rgb`.
    scalebar_loc : str, optional
        Location keyword for the scalebar annotation.
    flux_bar_decimals : int, optional
        Number of decimals on the flux-bar ticks.
    rgb_energies_existing : array-like or None, optional
        Energies/frequencies tied to the existing spectral bins (forwarded to `plot_rgb`).
    rgb_energies_target : array-like or None, optional
        Target RGB energies used during spectral conversion.
    rgb_log_spacing : bool, optional
        Assume log-spacing for implicit energies when converting spectral cubes.
    rgb_method : {"linear", "cubic"}, optional
        Interpolation scheme applied by `plot_rgb` during spectral conversion.
    verbose : bool, optional
        Emit diagnostic messages from `plot_rgb`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure that hosts the grid of images.
    axes : np.ndarray
        Array of matplotlib axes laid out in the grid.
    infos : list[dict]
        Per-panel metadata returned from `plot_rgb` (e.g., scaling thresholds).
    """
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("images must have shape (N, C, H, W) or (N, H, W, C)")

    channel_axis = int(channel_axis)
    if channel_axis < 0:
        channel_axis += images.ndim
    if not 0 <= channel_axis < images.ndim:
        raise ValueError(f"channel_axis={channel_axis} is out of bounds for images with ndim={images.ndim}")
    if channel_axis == 0:
        raise ValueError("channel_axis refers to the batch dimension; choose a different axis.")

    if channel_axis != 1:
        images = np.moveaxis(images, channel_axis, 1)

    if images.shape[1] < 3:
        raise ValueError("images must provide at least three channels for RGB conversion")

    N = images.shape[0]

    # Choose grid if not specified (near-square)
    if nrows is None and ncols is None:
        side = int(math.ceil(math.sqrt(N)))
        nrows, ncols = int(math.ceil(N / side)), side
        # Make it a bit wider than tall if that fits more naturally
        if (nrows - 1) * side >= N:
            nrows -= 1
    elif nrows is None:
        nrows = int(math.ceil(N / ncols))
    elif ncols is None:
        ncols = int(math.ceil(N / nrows))

    if figsize is None:
        figsize = (ncols * 3.0, nrows * 3.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    infos: List[dict] = []
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx < N:
                arr = images[idx]
                _, _, info = plot_rgb(
                    arr,
                    sat_min=sat_min,
                    sat_max=sat_max,
                    sigma=sigma,
                    log=log,
                    scale_mode=scale_mode,
                    show_flux_bars=show_flux_bars,
                    flux_bar_decimals=flux_bar_decimals,
                    scalebar_px=scalebar_px,
                    px_scale=px_scale,
                    scalebar_label=scalebar_label,
                    scalebar_loc=scalebar_loc,
                    rgb_energies_existing=rgb_energies_existing,
                    rgb_energies_target=rgb_energies_target,
                    rgb_log_spacing=rgb_log_spacing,
                    rgb_method=rgb_method,
                    verbose=verbose,
                    ax=ax,
                    name=None,          # do not save from inside
                )
                if titles is not None and idx < len(titles):
                    ax.set_title(titles[idx], fontsize=10)
                if share_axes:
                    ax.set_xticks([]); ax.set_yticks([])
                infos.append(info)
            else:
                # Hide unused cells
                ax.axis("off")
            idx += 1

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    display_plot_or_save(fig, filename=name, dpi=dpi, bbox_inches=bbox_inches)
    return fig, axes, infos


def _get_n_rows_from_n_samples(n_samples):
    """
    A function to get the number of rows from the given number of samples.

    Parameters:
    ----------
    n_samples: `int`.
    The number of samples.

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
    """2D Normal distribution"""
    const = 1 / (np.sqrt(2 * np.pi * sig ** 2))
    r = np.sqrt(x ** 2 + y ** 2)
    f = const * np.exp(-r ** 2 / (2 * sig ** 2))
    return f


def get_gaussian_kernel(domain, sigma):
    """"2D Gaussian kernel for fft convolution."""
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


def to_rgb_bands(
    cube,
    energies_existing=None,
    energies_target=None,
    *,
    log_spacing: bool = True,
    method: str = "linear",  # "linear" | "cubic" (Catmull–Rom)
):
    """
    Convert a spectral image cube (N, M, Q) into 3 RGB bands (3, M, Q) by
    interpolating along the spectral (channel) axis.

    Parameters
    ----------
    cube : np.ndarray or jnp.ndarray, shape (N, M, Q)
        Spectral image cube. Backend is inferred from this array (NumPy vs JAX).
    energies_existing : array-like or None
        The *log-space* energies/frequencies for each of the N channels.
        If None, they are assumed to be equally spaced in log space
        (linspace over [0, 1] with N points) if `log_spacing=True`,
        otherwise equally spaced in linear index space.
    energies_target : array-like of length 3 or None
        The *log-space* target energies to map to R, G, B. If None, three
        equidistant points (in the same space as `energies_existing`) are selected.
    log_spacing : bool, default True
        If `energies_existing` is None, interpret the channels as equally spaced
        in log space (True) or in linear index space (False).
    method : {"linear","cubic"}, default "linear"
        Interpolation method along the spectral axis.
        "cubic" uses a Catmull–Rom cubic Hermite spline (pure NumPy/JAX).

    Returns
    -------
    rgb : np.ndarray or jnp.ndarray, shape (3, M, Q)
        Interpolated RGB bands in the same backend as `cube`.

    Notes
    -----
    - Inputs in `energies_existing` and `energies_target` are expected to be in
      log-frequency units already (linear interpolation is performed in that space).
    - Extrapolation at the ends is clamped to endpoints for "linear".
      For "cubic", queries outside the range are computed via linear edge behavior.
    """
    # ---- choose backend from input ----
    is_jax = isinstance(cube, jnp.ndarray)
    xnp = jnp if is_jax else np

    if cube.ndim != 3:
        raise ValueError(f"`cube` must have shape (N, M, Q), got {cube.shape}.")

    N, M, Q = cube.shape
    if N < 2:
        raise ValueError("Need at least 2 spectral channels for interpolation.")
    if method not in ("linear", "cubic"):
        raise ValueError("method must be 'linear' or 'cubic'.")

    # ---- build/validate energies_existing (log space expected if provided) ----
    if energies_existing is None:
        if log_spacing:
            energies_existing = xnp.linspace(0.0, 1.0, N, dtype=xnp.float32)
        else:
            energies_existing = xnp.arange(N, dtype=xnp.float32)
    else:
        energies_existing = xnp.asarray(energies_existing, dtype=xnp.float32)
        if energies_existing.shape[0] != N:
            raise ValueError(
                f"energies_existing length {energies_existing.shape[0]} != N ({N})."
            )

    # Ensure ascending order for interp
    sort_idx = xnp.argsort(energies_existing)
    energies_existing = (
        xnp.take(energies_existing, sort_idx, axis=0) if is_jax else energies_existing[sort_idx]
    )
    cube = xnp.take(cube, sort_idx, axis=0) if is_jax else cube[sort_idx]

    # ---- choose/validate target energies ----
    if energies_target is None:
        emin = float(energies_existing[0])
        emax = float(energies_existing[-1])
        energies_target = xnp.linspace(emin, emax, 3, dtype=xnp.float32)
    else:
        energies_target = xnp.asarray(energies_target, dtype=xnp.float32)
        if energies_target.shape[0] != 3:
            raise ValueError("energies_target must have length 3 (for R, G, B).")

    # ---- helpers: linear vs cubic 1D interpolation for a single pixel spectrum ----
    def _interp_linear(vec_1d):
        return xnp.interp(energies_target, energies_existing, vec_1d)

    def _interp_cubic_catmull_rom(vec_1d):
        # Catmull–Rom cubic Hermite spline on monotone x with simple edge handling
        x = energies_existing
        y = vec_1d

        # Indices of the right bin edge for each query
        # For exact x[-1], searchsorted returns N, clip to N-1 later.
        j = xnp.searchsorted(x, energies_target, side="left")

        # For interior cubic, we need i-1, i, i+1, i+2 with i=j-1.
        # Clamp i to [1, N-3] so that (i-1) >= 0 and (i+2) <= N-1.
        i = xnp.clip(j - 1, 1, N - 3)

        # Gather supporting x/y
        x_im1 = x[i - 1]
        x_i   = x[i]
        x_ip1 = x[i + 1]
        x_ip2 = x[i + 2]

        y_im1 = y[i - 1]
        y_i   = y[i]
        y_ip1 = y[i + 1]
        y_ip2 = y[i + 2]

        # Local parameter t in [0,1]
        dx = (x_ip1 - x_i)
        # Avoid divide-by-zero for degenerate grids
        dx = xnp.where(dx == 0, xnp.finfo(xnp.float32).eps, dx)
        t = (energies_target - x_i) / dx

        # Tangents (finite-difference Catmull–Rom)
        m_i   = (y_ip1 - y_im1) / (x_ip1 - x_im1)
        m_ip1 = (y_ip2 - y_i)   / (x_ip2 - x_i)

        # Hermite basis
        t2 = t * t
        t3 = t2 * t
        h00 =  2.0 * t3 - 3.0 * t2 + 1.0
        h10 =        t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 =        t3 -       t2

        y_cubic = (
            h00 * y_i +
            h10 * dx * m_i +
            h01 * y_ip1 +
            h11 * dx * m_ip1
        )

        # Edge handling: for queries outside [x[0], x[-1]], fall back to linear clamp
        below = energies_target <= x[0]
        above = energies_target >= x[-1]
        # Two-point linear at edges
        y_lo = y[0] + (y[1] - y[0]) * (energies_target - x[0]) / (x[1] - x[0])
        y_hi = y[-2] + (y[-1] - y[-2]) * (energies_target - x[-2]) / (x[-1] - x[-2])
        y_out = xnp.where(below, y_lo, xnp.where(above, y_hi, y_cubic))
        return y_out

    interp_fn = _interp_linear if method == "linear" or N < 4 else _interp_cubic_catmull_rom

    # ---- interpolate along spectral axis for every pixel ----
    flat = cube.reshape(N, -1).T  # (P, N) where P = M*Q

    if is_jax:
        out_flat = vmap(interp_fn, in_axes=0)(flat)  # (P, 3)
    else:
        out_flat = xnp.stack([interp_fn(row) for row in flat], axis=0)

    # Reshape back to (M, Q, 3) then to (3, M, Q)
    out_spatial = out_flat.reshape(M, Q, 3)
    rgb = xnp.moveaxis(out_spatial, -1, 0)  # (3, M, Q)
    return rgb, energies_target
