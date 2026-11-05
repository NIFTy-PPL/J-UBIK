from dataclasses import asdict
from typing import Any, Union

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np

from ..parse.plotting import FieldPlottingConfig


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


def display_text(ax: plt.Axes, text: dict | str, **kwargs):
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
    sky_or_skies: np.ndarray | dict | None,
    data_model: jft.Model,
    data: np.ndarray,
    mask: np.ndarray,
    std: np.ndarray | float | None,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Sample-mean data model, the σ used, and the reduced chi² (per exposure
    for 3-D data) with its mean and std over the samples.

    With ``std`` None, ``data_model`` returns (prediction, inverse std), as
    `ShotNoiseGaussianLikelihoodBuilder.model` does: each sample's chi² uses
    its own σ and the returned σ is their mean. Each sample is evaluated once
    and summed over the unmasked pixels, so no per-sample image is kept.
    """
    if isinstance(std, float):
        std = np.full_like(data, std)

    def tree(x):
        while isinstance(x, jft.Vector):
            x = x.tree
        return x

    position_or_samples = tree(position_or_samples)
    sampled = (
        isinstance(position_or_samples, jft.Samples) and len(position_or_samples) > 0
    )
    if sampled:
        positions = (tree(si) for si in position_or_samples)
        skies = (
            [None] * len(position_or_samples) if sky_or_skies is None else sky_or_skies
        )
    else:
        if isinstance(position_or_samples, jft.Samples):
            position_or_samples = position_or_samples.pos
            # no samples: _get_model_samples_or_position gives [sky(pos)]
            if sky_or_skies is not None:
                (sky_or_skies,) = sky_or_skies
        positions, skies = [tree(position_or_samples)], [sky_or_skies]

    # per-exposure slices of the unmasked pixels, in data[mask] order
    bounds = np.cumsum([0, *mask.reshape(len(mask), -1).sum(1)])

    def redchi(model, sigma):
        """On the unmasked pixels; ``sigma`` None takes the fixed ``std``."""
        if len(mask.shape) == 2:
            ss = std[mask] if sigma is None else sigma
            return redchi2(data[mask], model, ss, model.size)
        # one exposure at a time, so no band-sized copy of data or std is made
        return [
            redchi2(
                data[e][mask[e]],
                model[a:b],
                std[e][mask[e]] if sigma is None else sigma[a:b],
                int(b - a),
            )
            for e, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))
        ]

    # sums over the unmasked pixels only; the images are built once at the end
    model_sum = sigma_sum = None
    redchis = []
    for x, sky in zip(positions, skies):
        prediction = data_model(x if sky is None else sky | x)
        sigma = None
        if std is None:
            prediction, inverse_std = prediction
            sigma = np.reciprocal(np.asarray(inverse_std, data.dtype))
            del inverse_std
        prediction = np.asarray(prediction, data.dtype)
        redchis.append(redchi(prediction, sigma))
        if model_sum is None:
            model_sum, sigma_sum = np.array(prediction), sigma
        else:
            model_sum += prediction
            if sigma is not None:
                sigma_sum += sigma
        del prediction, sigma
    model_sum /= len(redchis)
    model_mean = np.zeros_like(data)
    model_mean[mask] = model_sum
    del model_sum
    if std is None:
        sigma_sum /= len(redchis)
        std = np.zeros_like(data)
        std[mask] = sigma_sum
    del sigma_sum

    if sampled:
        redchi_mean, redchi_std = jft.mean_and_std(redchis)
    else:
        redchi_mean = redchis[0]
        redchi_std = np.full_like(redchi_mean, 0)
    return model_mean, std, (redchi_mean, redchi_std)
