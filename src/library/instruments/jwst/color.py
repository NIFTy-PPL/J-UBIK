from astropy import units as u
from astropy import constants as const


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
    def __init__(self, quantity: u.Quantity):
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

    def redshift(self, z: float):
        """
        Apply a redshift to the current color object.

        Parameters
        ----------
        z : float
            The redshift factor to apply.

        Returns
        -------
        Color
            A new Color object with the wavelength redshifted by
            a factor of (1 + z).

        Raises
        ------
        ValueError
            If the redshift factor `z` is negative.
        """
        return Color((1+z)*self.wavelength)

    def __repr__(self):
        """Returns a string representation of the Color object."""
        return f'Color: {self.energy}'


class ColorRange():
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

    def __contains__(self, item: Color):
        """Check if a given Color is within the range of this ColorRange."""
        assert isinstance(item, Color)
        return (
            (self.start.energy <= item.energy <= self.end.energy) or
            (self.start.energy >= item.energy >= self.end.energy)
        )

    def __repr__(self):
        """Returns a string representation of the ColorRange object."""
        return f'ColorRange: [{self.start.energy}, {self.end.energy}]'
