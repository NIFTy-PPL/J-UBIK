# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2026 Max-Planck-Society

"""Timed compatibility shims for renamed public spatial keys."""

import warnings
from collections.abc import Mapping

from ._spatial_validation import normalize_shape

SPATIAL_CONVENTIONS_URL = (
    "https://gitlab.mpcdf.mpg.de/ift/j-ubik/-/blob/main/docs/source/user/"
    "spatial-conventions.rst"
)

# `sdim` was replaced by `shape` in J-UBIK 0.4 (MR !238, commit b2dbb061).
# TODO Remove this module after 2026-12-17; `sdim` then raises again.
SDIM_REMOVAL_DATE = "2026-12-17"


def _sdim_to_shape(value, source: str) -> tuple[int, int]:
    """Warn about a deprecated ``sdim`` value and return it as ``(nx, ny)``."""

    nx, ny = normalize_shape(value, source)
    if nx != ny:
        # `sdim` never declared its axis order, so a rectangular pair cannot be
        # mapped to public (nx, ny) without guessing what the author meant.
        raise ValueError(
            f"{source}={value!r} is rectangular and its axis order is "
            "undefined; spell the grid out as `shape` in public (x, y) order. "
            f"See {SPATIAL_CONVENTIONS_URL}"
        )

    warnings.warn(
        f"{source} is deprecated since J-UBIK 0.4 (MR !238, commit b2dbb061); "
        "use `shape` in public (x, y) order. It stops working after "
        f"{SDIM_REMOVAL_DATE}. See {SPATIAL_CONVENTIONS_URL}",
        # FutureWarning, not DeprecationWarning: this is aimed at config
        # authors and pipeline scripts, which do not see DeprecationWarning
        # under the default filters.
        FutureWarning,
        stacklevel=3,
    )
    return (nx, ny)


def legacy_sdim_shape(grid_config: Mapping) -> tuple[int, int] | None:
    """Read a deprecated ``sdim`` grid key as a public ``(nx, ny)`` shape.

    Parameters
    ----------
    grid_config : Mapping
        Grid configuration, typically the ``grid`` block of a YAML config.

    Returns
    -------
    tuple[int, int] or None
        The shape to use in place of ``sdim``, or None if the key is absent.
    """

    if "sdim" not in grid_config:
        return None

    if "shape" in grid_config:
        raise ValueError(
            "grid config sets both the deprecated `sdim` and its replacement "
            f"`shape`; drop `sdim`. See {SPATIAL_CONVENTIONS_URL}"
        )

    return _sdim_to_shape(grid_config["sdim"], "grid `sdim`")


def legacy_sdim_argument(sdim, shape, *, caller: str) -> tuple[int, int] | None:
    """Read a deprecated ``sdim`` argument as a public ``(nx, ny)`` shape.

    Parameters
    ----------
    sdim : int or None
        Value passed to the deprecated keyword.
    shape : object
        Value passed to the replacement keyword, used to reject both at once.
    caller : str
        Name of the function whose keyword is deprecated, for the message.

    Returns
    -------
    tuple[int, int] or None
        The shape to use in place of ``sdim``, or None if ``sdim`` is None.
    """

    if sdim is None:
        return None

    if shape is not None:
        raise ValueError(
            f"{caller} got both the deprecated `sdim` and its replacement "
            f"`shape`; drop `sdim`. See {SPATIAL_CONVENTIONS_URL}"
        )

    return _sdim_to_shape(sdim, f"{caller} `sdim`")
