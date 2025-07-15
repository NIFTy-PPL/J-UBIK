# Copyright(C) 2025
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig,

from typing import Union

from nifty.re.correlated_field import (
    MaternAmplitude,
    NonParametricAmplitude,
    RegularCartesianGrid,
)
from nifty.re.num.stats_distributions import lognormal_prior, normal_prior

from .check_demands import check_demands


def _set_default_or_call(arg: Union[callable, tuple, list] | None, default: callable):
    """Either sets the default distribution or the callable"""
    if arg is None:
        return None

    if callable(arg):
        # TODO: do a check here that it is a valid distribution.
        return arg
    return default(*arg)


def assert_normalized_amplitude_model(
    model: MaternAmplitude | NonParametricAmplitude, name: str | None = None
) -> None:
    """This asserts that the model has no fluctuations or scale. Otherwise it will raise
    a ValueError. Furthermore, it checks that the model is either a `MaternAmplitude` or
    a `NonParametricAmplitude`.

    Parameters
    ----------
    model: MaternAmplitude | NonParametricAmplitude
        The model to be checked.
    name: str | None
        Additional string (model name) for the error message.
    """
    msg = (
        "Amplitude must be normalized."
        if name is None
        else f"{name} amplitude must be normalized."
    )
    msg += " It is not allowed to have `fluctuations`."

    if isinstance(model, MaternAmplitude):
        if model.scale is not None:
            raise ValueError(msg)
    elif isinstance(model, NonParametricAmplitude):
        if model.fluctuations is not None:
            raise ValueError(msg)
    else:
        raise ValueError(
            "The model must either be a `MaternAmplitude` or a `NonParametricAmplitude`."
        )


def build_normalized_amplitude_model(
    grid: RegularCartesianGrid,
    settings: dict | None,
    amplitude_model: str = "non_parametric",
    renormalize_amplitude: bool = True,
    prefix: str = None,
    kind: str = "amplitude",
) -> Union[None, MaternAmplitude, NonParametricAmplitude]:
    """
    Build an amplitude model based on
    the specified settings and model type.

    Parameters
    ----------
    grid: RegularCartesianGrid
        The grid on which the amplitude model is defined.
    settings: dict, optional
        A dictionary of settings that configure
        the amplitude model.
        If None, the builder returns None.
    amplitude_model: str, optional
        The type of amplitude model to build.
        Must be either "non_parametric" or "matern".
        Defaults to "non_parametric".
    renormalize_amplitude: bool, optional
        Whether to renormalize the amplitude in the "matern" model.
        Defaults to True.
    prefix: str, optional
        A prefix to add to the domain keys.
        Defaults to None.
    kind: str, optional
        A string to specify the kind of amplitude.
        Defaults to "amplitude".

    Returns
    -------
    amplitude: Amplitude or None
        A Model instance representing the amplitude model
        configured with the provided settings.
        Returns None if `settings` is None.

    Raises
    ------
    ValueError
        If `amplitude_model` is not "non_parametric" or "matern".
    """

    if settings is None:
        return None

    key = f"{prefix}_amplitude_" if prefix is not None else "amplitude_"

    if amplitude_model == "non_parametric":
        check_demands(
            key, settings, demands={"loglogavgslope", "flexibility", "asperity"}
        )
        return NonParametricAmplitude(
            grid,
            fluctuations=None,
            loglogavgslope=_set_default_or_call(
                settings["loglogavgslope"], normal_prior
            ),
            flexibility=_set_default_or_call(settings["flexibility"], lognormal_prior),
            asperity=_set_default_or_call(settings["asperity"], lognormal_prior),
            prefix=key,
        )
    elif amplitude_model == "matern":
        check_demands(key, settings, demands={"cutoff", "loglogslope"})
        return MaternAmplitude(
            grid,
            scale=None,
            cutoff=_set_default_or_call(settings["cutoff"], lognormal_prior),
            loglogslope=_set_default_or_call(settings["loglogslope"], normal_prior),
            renormalize_amplitude=renormalize_amplitude,
            kind=kind,
            prefix=key,
        )
    else:
        raise ValueError("Type must be 'non_parametric' or 'matern'.")
