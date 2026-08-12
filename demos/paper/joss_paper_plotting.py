import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import nifty.re as jft

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib import cm

from matplotlib import patches
from matplotlib.ticker import LogLocator

import jubik0 as ju


DEFAULT_SIZEBAR = dict(
    size_vertical=2,
    sep=4,
    color="white",
    frameon=False,
    fontproperties=fm.FontProperties(size=8),
    label_top=True,
    pad=0.5,
)

def joss_figsize(aspect_ratio=0.7):
    """
    Figsize helper

    Parameters
    ----------

    columns: 1 oder 2
    aspect_ratio: height/width
    """
    width_cm = 16.51        
    width_in = width_cm / 2.54
    height_in = width_in * aspect_ratio
    return (width_in, height_in)

def joss_cmap(cmap_name="magma", nan_color="black"):
    """
    Colormap helper
    """
    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color=nan_color)
    return cmap

# --- Colorbar Helper ---
def joss_colorbar(fig, im, label, ax=None, orientation='vertical', pad=0.05, fraction=0.05, labelpad=3):
    """
    Colorbar helper
    """
    if ax is None:
        ax = fig.gca()
    cbar = fig.colorbar(im, ax=ax, orientation=orientation, pad=pad, fraction=fraction)
    cbar.set_label(label, labelpad=labelpad)
    return cbar

def joss_scalebar(ax, length, label, loc="lower right", remove_ticks=True, **kwargs):
    """
    scalebar helper
    """
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    params = DEFAULT_SIZEBAR.copy()
    params.update(kwargs)
    scalebar = AnchoredSizeBar(ax.transData, length, label, loc, **params)
    ax.add_artist(scalebar)
    return scalebar