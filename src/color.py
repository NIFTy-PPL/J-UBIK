# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %%

import numpy as np
from astropy import units as u
from numpy.typing import NDArray


class Color(u.Quantity):
    """A class representing color based on wavelength, frequency, or energy. When
    iterable the vals are sorted by energy (frequency).

    Parameters
    ----------
    quantity : u.Quantity | Iterable[u.Quantity]
        A quantity with units of wavelength, frequency, or energy.

    Note
    ----
    One can set color ranges:
    - Consecutive: Color([1.0, 1.2, 1.5, 7.0] * u.keV)
    - Discontinuous: Color([[1.0, 1.2], [1.5, 7.0]] * u.keV)
    """

    def __new__(cls, quantity: u.Quantity):
        assert isinstance(quantity, u.Quantity), "Instantiate with units"
        assert len(quantity.shape) <= 2, "Only discontinuous ranges are supported"

        value = quantity.copy()
        if len(value.shape) > 0:
            value.sort(axis=0)  # For a 2D array, this sorts the rows.
        if len(value.shape) == 2:
            value.sort(axis=1)

        return super().__new__(cls, value=value)

    def redshift(self, z: float):
        """Corresponding color at redshift z.

        Parameters
        ----------
        z: The redshift

        Returns
        -------
        Redshifted Color: lambda(z) = lambda (1 + z).
        """
        return (1 + z) * self.to(u.Unit("Hz"), equivalencies=u.spectral())

    @property
    def center(self) -> u.Quantity:
        """Get the center of the colors."""
        if self.isscalar:
            return self

        if len(self.shape) == 1:
            return u.Quantity(
                [(self[ii + 1] + self[ii]) / 2 for ii in range(len(self) - 1)]
            )

        elif len(self.shape) == 2:
            # TODO : Maybe always transform color ranges to two-d arrays.
            bounds = get_2d_binbounds(self, self.unit) * self.unit
            return u.Quantity([((val[-1] + val[0]) / 2) for val in bounds])

        else:
            raise ValueError("Shouldn't end up here'")

    @staticmethod
    def _in_range(item: u.Quantity, rng: u.Quantity) -> bool:
        """Helper function for contains."""
        item = item.to(u.Unit("eV"), equivalencies=u.spectral()).value
        start = rng[0].to(u.Unit("eV"), equivalencies=u.spectral()).value
        end = rng[-1].to(u.Unit("eV"), equivalencies=u.spectral()).value

        return (start <= item) and (item <= end)

    def contains(self, quantity: u.Quantity) -> bool:
        """Check if the quantity is contained by the color-ranges."""
        if self.isscalar:
            raise ValueError(f"{self} is not a range")

        if len(self.shape) == 1:
            return self._in_range(quantity, self)

        elif len(self.shape) == 2:
            return any(self._in_range(quantity, rng) for rng in self)

        else:
            raise ValueError("Shouldn't end up here'")

    @property
    def is_continuous(self):
        if len(self.shape) == 1:
            return True

        end_points = self[:-1, 1]  # End of all bins except the last
        start_points = self[1:, 0]  # Start of all bins except the first
        return np.all(end_points == start_points)

    # TODO : Maybe merge with get_2d_binbounds?
    def binbounds(self, unit: u.Unit) -> "Color":
        """The binbounds of the binned color ranges in the requested unit.

        Note
        ----
        The color bins are assumed to be consecutive in energy. Hence, the
        minimum of each color range gets returned, the maximum of the binbounds
        is the and the maximum of the color range with the largest mean energy.
        """
        if self.isscalar:
            raise TypeError(
                "binbounds property requires an iterable "
                f"color-range, but we have {self.value}"
            )
        return self.to(unit, equivalencies=u.spectral())


def get_2d_binbounds(color: Color, unit: u.Unit) -> NDArray:
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

    range_2d = get_2d_binbounds(color_range, quantity.unit)

    indices = []
    for ii, rng in enumerate(range_2d):
        if Color(rng * quantity.unit).contains(quantity):
            indices.append(ii)

    return np.array(indices)
