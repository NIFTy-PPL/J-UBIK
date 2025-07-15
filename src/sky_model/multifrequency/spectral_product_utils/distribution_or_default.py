# Copyright(C) 2025
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig,

from typing import Any, Union

from nifty.re.correlated_field import WrappedCall


def build_distribution_or_default(
    arg: Union[callable, tuple, list],
    key: str,
    default: callable,
    shape: tuple = (),
    dtype: Any | None = None,
):
    """
    Build a distribution from an argument or use a default.

    Parameters
    ----------
    arg: Union[callable, tuple, list]
        The argument to be used for creating the distribution. If a callable
        is provided, it will be used directly. If a tuple or list is provided,
        the default callable will be used with the unpacked `arg` as its arguments.
    key: str
        The name or identifier for the distribution.
    default: callable
        The default callable to be used if `arg` is not a callable.
    shape: tuple, optional
        The shape of the resulting distribution. Defaults to an empty tuple `()`.
    dtype: type, optional
        The data type of the distribution. Defaults to None.

    Returns
    -------
    distribution: WrappedCall
        A WrappedCall instance representing the distribution with the specified
        `name`, `shape`, and `dtype`.
    """
    return WrappedCall(
        arg if callable(arg) else default(*arg),
        name=key,
        shape=shape,
        dtype=dtype,
        white_init=True,
    )
