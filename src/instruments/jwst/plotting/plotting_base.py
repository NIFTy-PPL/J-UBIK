from typing import Union, Any

import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np

from nifty8.re.library.mf_model import CorrelatedMultiFrequencySky

from ..parse.plotting import FieldPlottingConfig
from ..filter_projector import FilterProjector
from ..rotation_and_shift.coordinates_correction import CoordinatesWithCorrection


def plot_data_data_model_residuals(
    ims: list,
    axes: list,
    data_key: str,
    data: np.ndarray,
    data_model: np.ndarray,
    std: np.ndarray,
    plotting_config: FieldPlottingConfig,
):
    """Plot three panels (data, model, data-model).

    Parameters
    ----------

    """
    max_d = plotting_config.get_max(data)
    min_d = plotting_config.get_min(data)

    axes[0].set_title(f"Data {data_key}")
    axes[1].set_title("Data model")
    axes[2].set_title("Data - Data model")
    ims[0] = axes[0].imshow(
        data,
        norm=plotting_config.norm,
        vmin=min_d,
        vmax=max_d,
        **plotting_config.rendering,
    )
    ims[1] = axes[1].imshow(
        data_model,
        norm=plotting_config.norm,
        vmin=min_d,
        vmax=max_d,
        **plotting_config.rendering,
    )
    ims[2] = axes[2].imshow(
        (data - data_model) / std,
        vmin=-3,
        vmax=3,
        cmap="RdBu_r",
        **plotting_config.rendering,
    )

    return ims


def display_text(ax: plt.Axes, text: dict, **kwargs):
    """Display text on plot
    ax: matplotlib axis
    text: dict or str (default: {'s': str, 'color': 'white'})
    kwargs:
    - keyword: str
        options: 'top_left' (default), 'top_right', 'bottom_left', 'bottom_right'
    - x_offset_ticker: float (default: 0)
    - y_offset_ticker: float (default: 0)
    """
    keyword = kwargs.get("keyword", "top_left")
    x_offset_ticker = kwargs.get("x_offset_ticker", 0)
    y_offset_ticker = kwargs.get("y_offset_ticker", 0)

    if type(text) is str:
        text = dict(
            s=text,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

    if keyword == "top_left":
        ax.text(
            x=0.05 + x_offset_ticker * 0.05,
            y=0.95 - y_offset_ticker * 0.05,
            ha="left",
            va="top",
            transform=ax.transAxes,
            **text,
        )
    elif keyword == "top_right":
        ax.text(
            x=0.95 - x_offset_ticker * 0.05,
            y=0.95 - y_offset_ticker * 0.05,
            ha="right",
            va="top",
            transform=ax.transAxes,
            **text,
        )
    elif keyword == "bottom_left":
        ax.text(
            x=0.05 + x_offset_ticker * 0.05,
            y=0.05 + y_offset_ticker * 0.05,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            **text,
        )
    elif keyword == "bottom_right":
        ax.text(
            x=0.95 - x_offset_ticker * 0.05,
            y=0.05 + y_offset_ticker * 0.05,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            **text,
        )
    else:
        raise ValueError(
            "Invalid keyword. Use 'top_left', 'top_right', 'bottom_left', or 'bottom_right'."
        )


def chi2(data, model, std):
    """Computes the chi2 of the model compared to the data."""
    return np.nansum(((data - model) / std) ** 2)


def redchi2(data, model, std, dof):
    """Computes the reduced chi2 of the model compared to the data."""
    return chi2(data, model, std) / dof


def find_closest_factors(number):
    """
    Finds two integers whose multiplication is larger or equal to the input number,
    with the two output numbers being as close together as possible.

    Args:
        number: The input integer number.

    Returns:
        A tuple containing two integers (x, y) such that x * y >= number and the
        difference between x and y is minimized. If no such factors exist, returns
        None.
    """

    # Start with the square root of the number.
    ii = int(np.ceil(number**0.5))

    jj, kk = ii, ii

    jminus = kminus = 0
    while (jj - jminus) * (kk - kminus) >= number:
        if kminus == jminus:
            jminus += 1
        else:
            kminus += 1

    if jminus == kminus:
        return jj - jminus, kk - kminus + 1
    return jj - jminus + 1, kk - kminus


def get_position_or_samples_of_model(
    position_or_samples: Union[dict, jft.Samples],
    model: jft.Model,
    mean_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, std) of model"""

    if isinstance(position_or_samples, jft.Samples):
        mean, std = jft.mean_and_std([model(si) for si in position_or_samples])
    else:
        mean = model(position_or_samples)
        # FIXME: Can be handled by tree stuff
        if isinstance(mean, dict):
            std = {k: np.full(v.shape, 0) for k, v in mean.items()}
        else:
            std = np.full(mean.shape, 0.0)

    if mean_only:
        return mean

    return mean, std


def _determine_xpos(dkey: str):
    """Determine the x position of the panel in the residual plot"""
    index = int(dkey.split("_")[-1])
    return 3 + index


def _determine_ypos(
    dkey: str,
    filter_projector: FilterProjector,
) -> int:
    """Determine the y position of the panel in the residual plot.

    Parameters
    ----------
    dkey: str
        The data_key which will determine the energy bin and hence the position
        on the panel grid.
    filter_projector: FilterProjector
        The filter_projector of the reconstruction.
    ylen_offset: int
        An offset which corresponds to the bins not considered in determining
        the ypos of the panels. I.e. the sky can have more bins than the ones
        plotted in the panels, the ylen_offset corrects for this.

    Returns
    -------
    ypos: int
        The y-position on the panel grid.
    """
    tmp_dict = {}
    for ii, key in enumerate(filter_projector.keys_and_index):
        tmp_dict[key] = ii

    ekey = dkey.split("_")[0]
    return tmp_dict[ekey]


def _get_data_model_and_chi2(
    position_or_samples: Union[dict, jft.Samples, jft.Vector],
    sky_or_skies: np.ndarray,
    data_model: jft.Model,
    data: np.ndarray,
    mask: np.ndarray,
    std: np.ndarray,
):
    if isinstance(std, float):
        std = np.full_like(data, std)

    while isinstance(position_or_samples, jft.Vector):
        position_or_samples = position_or_samples.tree

    if isinstance(position_or_samples, jft.Samples):
        model_data = []
        for ii, si in enumerate(position_or_samples):
            tmp = np.zeros_like(data)
            while isinstance(si, jft.Vector):
                si = si.tree
            tmp[mask] = data_model(sky_or_skies[ii] | si)
            model_data.append(tmp)
        model_mean = jft.mean(model_data)
        redchi_mean, redchi_std = jft.mean_and_std(
            [
                redchi2(data[mask], m[mask], std[mask], data[mask].size)
                for m in model_data
            ]
        )

    else:
        model_data = np.zeros_like(data)
        position_or_samples = position_or_samples | sky_or_skies
        model_data[mask] = data_model(position_or_samples)
        redchi_mean = redchi2(data[mask], model_data[mask], std[mask], data[mask].size)
        redchi_std = 0
        model_mean = model_data

    return model_mean, (redchi_mean, redchi_std)


def _get_model_samples_or_position(position_or_samples, sky_model):
    if isinstance(position_or_samples, jft.Samples):
        return [sky_model(si) for si in position_or_samples]
    return sky_model(position_or_samples)


def determine_xlen_residuals(data_dict: dict, xmax_residuals):
    maximum = 0
    for dkey in data_dict.keys():
        index = int(dkey.split("_")[-1])
        if index > maximum:
            maximum = index
    maximum += 1
    if maximum > xmax_residuals:
        return xmax_residuals
    return maximum  # because 0 is already there and will not be counted


def get_alpha_and_reference(light_model):
    from charm_lensing.physical_models.hybrid_model import HybridModel

    light_model: HybridModel = light_model
    model: CorrelatedMultiFrequencySky | Any = light_model.nonparametric

    if isinstance(model, CorrelatedMultiFrequencySky):
        alpha = model.spectral_index_distribution
        reference = model.reference_frequency_distribution
        return alpha, reference

    def nothing(_):
        return np.zeros((12, 12))

    return nothing, nothing


def get_shift_rotation_correction(
    position_or_samples: Union[dict, jft.Samples],
    correction_model: CoordinatesWithCorrection | None,
):
    if not isinstance(correction_model, CoordinatesWithCorrection):
        return (0, 0), (0, 0), 0, 0

    shift_mean, shift_std = get_position_or_samples_of_model(
        position_or_samples, correction_model.shift_prior
    )
    rotation_mean, rotation_std = get_position_or_samples_of_model(
        position_or_samples, correction_model.rotation_prior
    )
    shift_mean, shift_std = shift_mean.reshape(2), shift_std.reshape(2)
    rotation_mean, rotation_std = rotation_mean[0], rotation_std[0]

    return (shift_mean, shift_std), (rotation_mean, rotation_std)
