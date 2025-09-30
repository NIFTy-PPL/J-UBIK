# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %%

from astropy import units as u
from numpy import argsort


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

    def center(self) -> u.Quantity:
        """Get the center of the colors."""
        if self.isscalar:
            return self

        if len(self.shape) == 1:
            return u.Quantity(
                [(self[ii + 1] + self[ii]) / 2 for ii in range(len(self) - 1)]
            )

        elif len(self.shape) == 2:
            return u.Quantity([((val[-1] + val[0]) / 2) for val in self])

        else:
            raise ValueError("Shouldn't end up here'")

    @staticmethod
    def _in_range(item: u.Quantity, rng: u.Quantity) -> bool:
        """Helper function for contains."""
        item = item.to(u.Unit("eV"), equivalencies=u.spectral())
        start: u.Quantity = rng[0].to(u.Unit("eV"), equivalencies=u.spectral())
        end: u.Quantity = rng[-1].to(u.Unit("eV"), equivalencies=u.spectral())

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

    def binbounds(self, unit: u.Unit):
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
