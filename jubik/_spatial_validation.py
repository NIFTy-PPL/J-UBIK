# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2026 Max-Planck-Society

"""Shared normalization for config parsing and spatial geometry construction."""

import numbers

import numpy as np
from astropy import units as u


def normalize_shape(value, name: str) -> tuple[int, int]:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        values = (value, value)
    else:
        try:
            values = tuple(value)
        except TypeError:
            raise ValueError(f"{name} must be an int or a pair of ints, got {value!r}")
    if len(values) != 2:
        raise ValueError(f"{name} must contain two entries, got {value!r}")
    for v in values:
        if not isinstance(v, numbers.Integral) or isinstance(v, bool):
            raise ValueError(f"{name} must contain integers, got {value!r}")
        if v <= 0:
            raise ValueError(f"{name} must contain two positive integers, got {value!r}")
    return tuple(int(v) for v in values)


def normalize_fov(value, name: str) -> u.Quantity:
    if isinstance(value, (list, tuple)):
        quantity = u.Quantity([u.Quantity(v) for v in value])
    else:
        quantity = u.Quantity(value)
    if quantity.isscalar:
        quantity = u.Quantity((quantity, quantity))
    if quantity.shape != (2,):
        raise ValueError(f"{name} must contain two angular sizes, got {value!r}")
    if not quantity.unit.is_equivalent(u.rad):
        raise u.UnitConversionError(f"{name} must carry angular units, got {value!r}")
    if np.any(quantity <= 0 * quantity.unit):
        raise ValueError(f"{name} must contain two positive angular sizes, got {value!r}")
    quantity = quantity.copy()
    quantity.flags.writeable = False
    return quantity
