# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %%

import numpy as np
from astropy import units as u
from numpy.typing import NDArray


class Color(u.Quantity):
    """A class representing color based on wavelength, frequency, or energy. A color
    range gets sorted.

    Parameters
    ----------
    quantity : u.Quantity | Iterable[u.Quantity]
        A quantity with units of wavelength, frequency, or energy.

    Note
    ----
    1. One can set color ranges:
       - Consecutive: Color([1.0, 1.2, 1.5, 7.0] * u.keV)
       - Discontinuous: Color([[1.0, 1.2], [1.5, 7.0]] * u.keV)

    2. Ranges are internally always represented by two dimensional arrays. See
       `get_2d_binbounds`.
    """

    def __new__(cls, quantity: u.Quantity) -> "Color":
        assert isinstance(quantity, u.Quantity), "Instantiate with units"
        assert len(quantity.shape) <= 2, "Only discontinuous ranges are supported"

        value = quantity.copy()
        if not value.isscalar:
            value.sort(axis=0)  # Sort the ranges
            if len(value.shape) == 2:
                value.sort(axis=1)  # Sort in ranges
            value = value.unit * get_2d_binbounds(value, value.unit)

        return super().__new__(cls, value=value)

    def redshift(self, z: float) -> "Color":
        """Corresponding color at redshift z.

        Parameters
        ----------
        z: The redshift

        Returns
        -------
        Redshifted Color: nu(z) = nu / (1 + z).
        Redshifted Color: lambda(z) = lambda (1 + z).
        """
        return self.to(u.Unit("Hz"), equivalencies=u.spectral()) / (1 + z)

    def __contains__(self, key) -> bool:
        raise NotImplementedError("Temporary to check for old dependencies")
        # return self.contains(key)

    @property
    def center(self) -> u.Quantity:
        """Get the center of the colors."""
        if self.isscalar:
            return self

        elif len(self.shape) == 2:
            if not self.shape[1] == 2:
                raise ValueError("Ranges should always be represented by (N, 2) shapes")

            return u.Quantity([((val[-1] + val[0]) / 2) for val in self])

        else:
            raise ValueError("Shouldn't end up here'")

    def _in_range(self, rng: u.Quantity, quantity: u.Quantity) -> bool:
        """Helper function for contains."""
        assert len(rng.shape) == 1
        assert rng.shape[0] == 2

        quantity = quantity.to(self.unit, equivalencies=u.spectral()).value
        return (rng[0].value <= quantity) and (quantity <= rng[-1].value)

    def contains(self, quantity: u.Quantity) -> bool:
        """Check if the quantity is contained by the color-ranges."""
        if self.isscalar:
            raise ValueError(f"{self} is not a range")

        if len(self.shape) == 1:
            if not self.shape[0] == 2:
                raise ValueError("Ranges should always be represented by shape 2")
            return self._in_range(self, quantity)

        elif len(self.shape) == 2:
            if not self.shape[1] == 2:
                raise ValueError("Ranges should always be represented by (N, 2) shapes")

            return any(self._in_range(rng, quantity) for rng in self)

        else:
            raise ValueError("Shouldn't end up here")

    @property
    def is_continuous(self) -> bool:
        if self.isscalar:
            raise ValueError("Scalar Value can't be continuous")

        if len(self.shape) == 1:
            return True

        start_points = self[1:, 0]  # Start of all bins except the first
        end_points = self[:-1, 1]  # End of all bins except the last
        return np.all(end_points == start_points)


def get_2d_binbounds(color: u.Quantity, unit: u.Unit) -> NDArray:
    """Transform the color range into a two dimensional array.

    Parameters
    ----------
    color: Color
    unit: astropy.units.Unit

    NOTE
    ----
    [1.2, 2.4, 5]      -> [[1.2, 2.4],
                           [2.4, 5.0]]

    [[10., 15., 18.],  -> [[10., 15.],
     [50., 56., 58.]]      [15., 18.],
                           [50., 56.],
                           [56., 58.]]
    """

    if color.isscalar:
        raise ValueError("Only spectral ranges can be considered")

    values_in_unit = color.to(unit, equivalencies=u.spectral()).value
    if len(color.shape) == 1:
        return np.stack((values_in_unit[:-1], values_in_unit[1:]), axis=1)

    elif len(color.shape) == 2:
        left_bounds = values_in_unit[:, :-1]
        right_bounds = values_in_unit[:, 1:]

        # NOTE: Stack them along a new third dimension to create pairs.
        # The result is a 3D array of shape (N, M-1, 2).
        stacked_bins = np.stack([left_bounds, right_bounds], axis=2)
        return stacked_bins.reshape(-1, 2)

    else:
        # This case is prevented by the assertion in Color.__new__
        # but is included for completeness.
        raise ValueError(f"Unsupported shape for Color object: {color.shape}")


def get_spectral_range_index(color_range: Color, quantity: u.Quantity | Color):
    """Get the index of the `quantity` inside the `color_range`.

    Parameters
    ----------
    color_range: Color
        The range of colors to be indexed.
    quantity: u.Quantity | Color
        The spectral quantity to find the index inside the color_range.
    """

    if color_range.isscalar:
        raise ValueError("color_range must be a range")

    return np.array(
        [ii for ii, rng in enumerate(color_range) if rng.contains(quantity)]
    )
