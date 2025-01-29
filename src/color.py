# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from astropy import units as u, constants as const
from numpy import argsort

from typing import Optional


class Color(u.Quantity):
    """A class representing color based on wavelength, frequency, or energy.

    The Color class allows conversion between wavelength, frequency, and energy
    representations using physical constants from astropy.

    Parameters
    ----------
    quantity : u.Quantity
        A quantity with units of wavelength, frequency, or energy.
        The input must have a physical type of 'length', 'frequency',
        or 'energy'.

    Raises
    ------
    IOError
        If the input quantity does not have units (is not a u.Quantity).
    ValueError
        If the input quantity has an unsupported physical type.
    """

    def __init__(
        self,
        quantity: u.Quantity,
    ):
        if not isinstance(quantity, u.Quantity):
            raise IOError('Instantiate with a quantity that has units')

        if quantity.unit.physical_type == 'length':
            self._init_length(quantity)
        elif quantity.unit.physical_type == 'frequency':
            self._init_frequency(quantity)
        elif quantity.unit.physical_type == 'energy':
            self._init_energy(quantity)
        else:
            raise ValueError(
                f"Unsupported physical type: {quantity.physical_type}")

    def _init_length(self, wavelength: u.Quantity):
        # Initialization using wavelength
        self.wavelength = wavelength.to(u.m)
        self.frequency = (const.c / wavelength).to(u.Hz)
        self.energy = (const.h * self.frequency).to(u.eV)

    def _init_frequency(self, frequency: u.Quantity):
        # Initialization using frequency
        self.frequency = frequency.to(u.Hz)
        self.wavelength = (const.c / frequency).to(u.m)
        self.energy = (const.h * frequency).to(u.eV)

    def _init_energy(self, energy: u.Quantity):
        # Initialization using energy
        self.energy = energy.to(u.eV)
        self.frequency = (energy / const.h).to(u.Hz)
        self.wavelength = (const.c / self.frequency).to(u.m)

    def to_unit(self, unit: u.Unit):
        if unit.physical_type == 'length':
            value = self.wavelength
        elif unit.physical_type == 'frequency':
            value = self.frequency
        elif unit.physical_type == 'energy':
            value = self.energy
        return value.to(unit)

    def redshift(self, z: float):
        """
        Corresponding color at redshift z.

        Parameters
        ----------
        z: The redshift

        Returns
        -------
        Redshifted Color: lambda(z) = lambda (1 + z).

        """
        return Color((1+z)*self.wavelength)

    def __repr__(self):
        return f'Color({self.energy})'


class ColorRange:
    """
    A class representing a range of colors, defined by start and
    end Color objects.

    Parameters
    ----------
    start : Color
        The starting Color object of the range.
    end : Color
        The ending Color object of the range.

    Raises
    ------
    AssertionError
        If `start` or `end` are not instances of the Color class.
    """

    def __init__(self, start: Color, end: Color):
        """Initialize the ColorRange object."""
        assert isinstance(start, Color) and isinstance(end, Color)
        self.start = start
        self.end = end

    @ property
    def center(self):
        '''Returns the center/mean of each color range.'''
        return Color((self.end.energy - self.start.energy) / 2 + self.start.energy)

    def __contains__(self, item: Color):
        """Check if a given Color is within the range of this ColorRange."""
        assert isinstance(item, Color)
        return (
            (self.start.energy <= item.energy <= self.end.energy) or
            (self.start.energy >= item.energy >= self.end.energy)
        )

    def __repr__(self):
        """Returns a string representation of the ColorRange object."""
        return f'ColorRange([{self.start.energy}, {self.end.energy}])'

    def to_unit(self, unit: u.Unit) -> tuple[u.Quantity]:
        return self.start.to_unit(unit), self.end.to_unit(unit)

    @ property
    def binbounds(self):
        '''The binbounds of the binned color range.

        Note
        ----
        The bins are assumed to be consecutive. Hence, the minimum of each
        color range gets returned, the maximum of the binbounds is the and the
        maximum of the color range with the largest mean energy. '''
        return [self.start, self.end]

    def binbounds_in(self, unit: u.Unit):
        '''The binbounds of the binned color ranges in the requested unit.

        Note
        ----
        The color bins are assumed to be consecutive in energy. Hence, the
        minimum of each color range gets returned, the maximum of the binbounds
        is the and the maximum of the color range with the largest mean energy.
        '''
        unit_type = {
            u.physical.length: 'wavelength',
            u.physical.frequency: 'frequency',
            u.physical.energy: 'energy'
        }[unit.physical_type]

        return [getattr(bb, unit_type).to(unit).value for bb in self.binbounds]


class ColorRanges:
    """
    A class representing multiple bins of colors, i.e. a set of color ranges.

    Parameters
    ----------
    color_ranges:
        The color ranges that the BinnedColorRanges should hold.

    Raises
    ------
    AssertionError
        If any of the colorr are not instances of the ColorRange class.
    """

    def __init__(self, color_ranges: list[ColorRange]):
        for cr in color_ranges:
            assert isinstance(cr, ColorRange)

        sortid = argsort([cr.center.value for cr in color_ranges])
        self.color_ranges = [color_ranges[ii] for ii in sortid]

    @ property
    def binbounds(self):
        '''The binbounds (color) of the binned color ranges.

        Note
        ----
        The bins are assumed to be consecutive. Hence, the minimum of each
        color range gets returned, the maximum of the binbounds is the and the
        maximum of the color range with the largest mean energy. '''
        return [cr.start for cr in self.color_ranges] + [self.color_ranges[-1].end]

    def binbounds_in(self, unit: u.Unit):
        '''The binbounds of the binned color ranges in the requested unit.

        Note
        ----
        The color bins are assumed to be consecutive in energy. Hence, the
        minimum of each color range gets returned, the maximum of the binbounds
        is the and the maximum of the color range with the largest mean energy.
        '''
        unit_type = {
            u.physical.length: 'wavelength',
            u.physical.frequency: 'frequency',
            u.physical.energy: 'energy'
        }[unit.physical_type]

        return [getattr(bb, unit_type).to(unit).value for bb in self.binbounds]

    @ property
    def centers(self):
        '''Returns the centers/mean of each color range.'''
        return [cr.center for cr in self.color_ranges]

    @property
    def shape(self):
        return (len(self.color_ranges),)

    def get_color_range_from_color(self, color: Color):
        '''Returns all color ranges which contain the color. Note: These could
        be more then one if the color ranges are not disjoint.'''
        assert isinstance(color, Color), f'{color} must be a Color'
        assert color in self, f'{color} is not in {self}'
        return [ii for ii, cr in enumerate(self.color_ranges) if color in cr]

    def __repr__(self):
        crs = ''
        for ii, cr in enumerate(self.color_ranges):
            crs += f'\n{ii}: {cr.__repr__()}'
        return f'ColorRanges({crs})'

    def __contains__(self, item: Color):
        return any([item in cr for cr in self.color_ranges])

    def __len__(self):
        return len(self.color_ranges)

    def __getitem__(self, index: int):
        '''Get color range from bin index.'''
        return self.color_ranges[index]

    def to_unit(self, unit: u.Unit) -> list[tuple[u.Quantity]]:
        crs = []
        for ii, cr in enumerate(self.color_ranges):
            crs.append(cr.to_unit(unit))
        return crs
