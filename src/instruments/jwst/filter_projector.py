# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

import nifty8.re as jft


class FilterProjector(jft.Model):
    """
    A nifty8.re.Model that projects input data into specified filters
    defined by color keys.

    The FilterProjector class takes a sky domain and a mapping between keys
    and colors, and applies a projection of input data according to the filters.
    It supports querying keys based on colors and efficiently applies
    transformations for multi-channel inputs.
    """

    def __init__(self, sky_domain: jft.ShapeWithDtype, keys_and_colors: dict):
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
        """
        self.keys_and_colors = keys_and_colors
        self.keys_and_index = {
            key: index for index, key in enumerate(keys_and_colors.keys())
        }

        self.apply = self._get_apply()
        super().__init__(domain=sky_domain)

    def get_key(self, color):
        """Returns the key that corresponds to the given color."""
        out_key = ""
        for k, v in self.keys_and_colors.items():
            if color in v:
                if out_key != "":
                    raise IndexError(
                        f"{color} fits into multiple keys of the " "FilterProjector"
                    )
                out_key = k
        if out_key == "":
            raise IndexError(
                f"{color} doesn't fit in the bounds of the FilterProjector."
            )

        return out_key

    def _get_apply(self):
        """Returns a function that applies the projection to a given input."""
        if len(self.keys_and_index) == 1:
            key, _ = next(iter(self.keys_and_index.items()))
            return lambda x: {key: x}
        else:
            return lambda x: {
                key: x[index] for key, index in self.keys_and_index.items()
            }

    def __call__(self, x):
        return self.apply(x)
