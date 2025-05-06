import matplotlib.pyplot as plt
import numpy as np
from jubik0.wcs.wcs_base import WcsMixin
import types


def plot_sky_coords(
    index,
    ax,
    sky_coords,
    marker="o",
    marker_color="red",
    marker_size=50,
    alpha=1.0,  # Added alpha parameter
    labels=None,
    sky_coords_behavior_by_index: callable = lambda index, sky_coords: sky_coords,
):
    """
    Plot SkyCoord points on an existing axes.

    Parameters
    ----------
    index : int
        Behaviour according to the `sky_coords_behavior_by_index`.
    ax : matplotlib.axes.Axes
        The axes object to plot coordinates on
    sky_coords : SkyCoord or list of SkyCoord
        SkyCoord object(s) to plot as points
    marker : str, optional
        Marker style for SkyCoord points
    marker_color : str or list, optional
        Color(s) for the markers
    marker_size : float, optional
        Size of the markers
    alpha : float, optional
        Transparency level of the markers (0.0 to 1.0, where 0 is fully transparent)
    labels : list, optional
        Labels for the SkyCoord points
    sky_coords_behavior_by_index: callable[index, sky_coords] -> sky_coords
    """
    if sky_coords is None:
        return

    # Handle both single SkyCoord and lists of SkyCoord
    coords_list = sky_coords if isinstance(sky_coords, list) else [sky_coords]
    colors = (
        marker_color
        if isinstance(marker_color, list)
        else [marker_color] * len(coords_list)
    )

    for j, coord in enumerate(sky_coords_behavior_by_index(index, coords_list)):
        # Plot the coordinate directly with SkyCoord support
        ax.scatter(
            coord.ra.deg,
            coord.dec.deg,
            transform=ax.get_transform("icrs"),
            marker=marker,
            color=colors[j % len(colors)],
            s=marker_size,
            alpha=alpha,  # Apply alpha for transparency
        )

        # Add labels if provided
        if labels is not None and j < len(labels):
            ax.text(
                coord.ra.deg,
                coord.dec.deg,
                labels[j],
                transform=ax.get_transform("icrs"),
                color=colors[j % len(colors)],
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, pad=2),
                alpha=alpha,  # Apply alpha to text as well
            )


def plot_jwst_panels(
    data_list,
    wcs_list,
    nrows,
    ncols,
    figsize=(10, 10),
    vmin=None,
    vmax=None,
    coords_plotter: callable = lambda ii, ax: None,  # Function to plot coordinates
):
    """
    Plot JWST data panels with optional SkyCoord points overlaid.

    Parameters
    ----------
    data_list : list
        List of data arrays to display
    wcs_list : list
        List of WCS objects corresponding to data_list
    nrows, ncols : int
        Number of rows and columns for subplot layout
    figsize : tuple, optional
        Figure size in inches (width, height)
    vmin, vmax : float, optional
        Minimum and maximum values for colormap scaling
    coords_plotter : callable
        Function that accepts an index and axes object as input and plots coordinates
        Can be created using functools.partial(plot_sky_coords, sky_coords=some_coords)
    """
    fig = plt.figure(figsize=figsize)
    axes = []

    for i in range(len(data_list)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=wcs_list[i])
        axes.append(ax)

        im = ax.imshow(
            data_list[i], origin="lower", aspect="equal", vmin=vmin, vmax=vmax
        )
        overlay = ax.get_coords_overlay("icrs")
        overlay.grid(color="white", ls="dotted")
        overlay[0].set_axislabel("Right Ascension (icrs)")
        overlay[1].set_axislabel("Declination (icrs)")

        coords_plotter(i, ax)

    return fig, axes
