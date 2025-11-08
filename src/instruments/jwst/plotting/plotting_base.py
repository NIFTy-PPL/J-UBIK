from dataclasses import asdict
from typing import Any, Union

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np

from ....sky_model.multifrequency.spectral_product_mf_sky import SpectralProductSky
from ..parse.plotting import FieldPlottingConfig
from ..rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectedShiftAndRotation,
    CoordinatesCorrectedShiftOnly,
)


def plot_data_data_model_residuals(
    ims: list,
    axes: list,
    data_key: str,
    data: np.ndarray,
    data_model: np.ndarray,
    std: np.ndarray,
    residual_over_std: bool,
    residual_config: FieldPlottingConfig,
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
    if residual_over_std:
        axes[2].set_title("(Data - Data model) / std")
    else:
        axes[2].set_title("Data - Data model")
        std = 1.0

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
        **asdict(residual_config),
        **residual_config.rendering,
    )

    return ims


def plot_data_residuals(
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
    axes[1].set_title("Data - model / std")
    ims[0] = axes[0].imshow(
        data,
        norm=plotting_config.norm,
        vmin=min_d,
        vmax=max_d,
        **plotting_config.rendering,
    )
    ims[1] = axes[1].imshow(
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


def _get_model_samples_or_position(position_or_samples, model):
    if isinstance(position_or_samples, jft.Samples):
        if len(position_or_samples) == 0:
            return np.array([model(position_or_samples.pos)])
        return np.array([model(si) for si in position_or_samples])
    return model(position_or_samples)


def _get_std_from_inversestdmodel(
    position_or_samples: Union[dict, jft.Samples, jft.Vector],
    inverse_std: jft.Model,
):
    if isinstance(position_or_samples, jft.Samples) and len(position_or_samples) > 0:
        return jft.mean([1 / inverse_std(x) for x in position_or_samples])
    elif isinstance(position_or_samples, jft.Samples):
        position_or_samples = position_or_samples.pos

    return 1 / inverse_std(position_or_samples)


def _get_data_model_and_chi2(
    position_or_samples: Union[dict, jft.Samples, jft.Vector],
    sky_or_skies: np.ndarray | None,
    data_model: jft.Model,
    data: np.ndarray,
    mask: np.ndarray,
    std: Union[np.ndarray, jft.Model],
) -> tuple[np.ndarray, tuple[float, float]]:
    if isinstance(std, float):
        std = np.full_like(data, std)

    while isinstance(position_or_samples, jft.Vector):
        position_or_samples = position_or_samples.tree

    if isinstance(position_or_samples, jft.Samples) and len(position_or_samples) > 0:
        model_d = []
        for ii, si in enumerate(position_or_samples):
            tmp = np.zeros_like(data)
            while isinstance(si, jft.Vector):
                si = si.tree
            if sky_or_skies is None:
                tmp[mask] = data_model(si)
            else:
                tmp[mask] = data_model(sky_or_skies[ii] | si)
            model_d.append(tmp)

        model_mean = jft.mean(model_d)

        if len(mask.shape) == 2:
            redchi_mean, redchi_std = jft.mean_and_std(
                [
                    redchi2(data[mask], model[mask], std[mask], data[mask].size)
                    for model in model_d
                ]
            )
        else:
            redchi_mean, redchi_std = [], []
            for ii, (dd, mm, ss) in enumerate(zip(data, mask, std)):
                rchi_mean, rchi_std = jft.mean_and_std(
                    [
                        redchi2(dd[mm], model[ii][mm], ss[mm], dd[mm].size)
                        for model in model_d
                    ]
                )
                redchi_mean.append(rchi_mean)
                redchi_std.append(rchi_std)

    else:
        if isinstance(position_or_samples, jft.Samples):
            position_or_samples = position_or_samples.pos

        model_d = np.zeros_like(data)
        if sky_or_skies is not None:
            position_or_samples = position_or_samples | sky_or_skies
        model_d[mask] = data_model(position_or_samples)
        if len(mask.shape) == 2:
            redchi_mean = redchi2(data[mask], model_d[mask], std[mask], data[mask].size)
        else:
            redchi_mean = []
            for ii, (dd, mo, mm, ss) in enumerate(zip(data, model_d, mask, std)):
                redchi_mean.append(redchi2(dd[mm], mo[mm], ss[mm], dd[mm].size))

        redchi_std = np.full_like(redchi_mean, 0)
        model_mean = model_d

    return np.array(model_mean), (redchi_mean, redchi_std)


def get_alpha_and_reference(light_model):
    from charm_lensing.physical_models.hybrid_model import HybridModel
    from charm_lensing.physical_models.multifrequency_models.vstack_model import (
        VstackModel,
    )

    light_model: HybridModel = light_model
    if isinstance(light_model, HybridModel):
        model: SpectralProductSky | Any = light_model.nonparametric
    elif isinstance(light_model, VstackModel):
        model = light_model.infrared.nonparametric

    if isinstance(model, SpectralProductSky):
        alpha = model.spectral_index_distribution
        reference = model.reference_frequency_distribution
        return alpha, reference

    def nothing(_):
        return np.zeros((12, 12))

    return nothing, nothing


def get_shift_rotation_correction(
    position_or_samples: Union[dict, jft.Samples],
    correction_model: Union[
        CoordinatesCorrectedShiftOnly, CoordinatesCorrectedShiftAndRotation
    ]
    | None,
):
    if (not isinstance(correction_model, CoordinatesCorrectedShiftOnly)) or (
        not isinstance(correction_model, CoordinatesCorrectedShiftAndRotation)
    ):
        return (0, 0), (0, 0), 0, 0

    shift_mean, shift_std = get_position_or_samples_of_model(
        position_or_samples, correction_model.shift_and_rotation.shift
    )
    rotation_mean, rotation_std = get_position_or_samples_of_model(
        position_or_samples, correction_model.shift_and_rotation.rotation_angle
    )
    shift_mean, shift_std = shift_mean.reshape(2), shift_std.reshape(2)
    rotation_mean, rotation_std = rotation_mean[0], rotation_std[0]

    return (shift_mean, shift_std), (rotation_mean, rotation_std)
