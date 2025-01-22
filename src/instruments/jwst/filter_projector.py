# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
import nifty8.re as jft

import numpy as np
from typing import Optional, Union


def _sorted_keys_and_index(keys_and_colors: dict):
    keys, colors = keys_and_colors.keys(), keys_and_colors.values()
    sorted_indices = np.argsort([c.center.energy.value for c in colors])
    return {key: index for key, index in zip(keys, sorted_indices)}


class FilterProjector(jft.Model):
    """
    A nifty8.re.Model that projects input data into specified filters
    defined by color keys.

    The FilterProjector class takes a sky domain and a mapping between keys
    and colors, and applies a projection of input data according to the filters.
    It supports querying keys based on colors and efficiently applies
    transformations for multi-channel inputs.
    """

    def __init__(
        self,
        sky_domain: Union[jft.ShapeWithDtype, dict[str, jft.ShapeWithDtype]],
        keys_and_colors: dict,
        keys_and_index: Optional[dict],
        sky_key: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        sky_domain : jft.ShapeWithDtype
            The domain for the sky data, defining the shape and data type of
            the input.
        keys_and_colors : dict
            A dictionary where the keys are filter names (or keys) and the
            values are lists of colors associated with each filter.
            This defines how inputs will be mapped to the respective filters.
        keys_and_index : Optional[dict]
            A dictionary holding the filter names as keys and the associated
            index in the reconstruction grid.
        sky_key : Optional[str]
            If a sky_key is provided the sky-array gets unwrapped in the call.
        """
        if sky_key is None:
            assert len(sky_domain.shape) == 3, (
                'FilterProjector expects a sky with 3 dimensions.')
        else:
            assert len(sky_domain[sky_key].shape) == 3, (
                'FilterProjector expects a sky with 3 dimensions.')

        self._sky_key = sky_key
        self.keys_and_colors = keys_and_colors
        self.keys_and_index = keys_and_index
        if keys_and_index is None:
            self.keys_and_index = _sorted_keys_and_index(keys_and_colors)

        super().__init__(domain=sky_domain)

    def get_key(self, color):
        """Returns the key that corresponds to the given color."""
        out_key = ''
        for k, v in self.keys_and_colors.items():
            if color in v:
                if out_key != '':
                    raise IndexError(
                        f'{color} fits into multiple keys of the '
                        'FilterProjector')
                out_key = k
        if out_key == '':
            raise IndexError(
                f"{color} doesn't fit in the bounds of the FilterProjector.")

        return out_key

    def __call__(self, x):
        if self._sky_key is not None:
            x = x[self._sky_key]
        return {key: x[index] for key, index in self.keys_and_index.items()}
