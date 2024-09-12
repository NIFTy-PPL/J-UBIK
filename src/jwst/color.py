from astropy import units as u
from astropy import constants as const


class Color(u.Quantity):
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
        return Color((1+z)*self.wavelength)

    def __repr__(self):
        return f'Color: {self.energy}'


class ColorRange():
    def __init__(self, start: Color, end: Color):
        assert isinstance(start, Color) and isinstance(end, Color)
        self.start = start
        self.end = end

    @property
    def center(self):
        return Color((self.end.energy - self.start.energy) / 2 + self.start.energy)

    def __contains__(self, item: Color):
        assert isinstance(item, Color)
        return (
            (self.start.energy <= item.energy <= self.end.energy) or
            (self.start.energy >= item.energy >= self.end.energy)
        )

    def __repr__(self):
        return f'ColorRange: [{self.start.energy}, {self.end.energy}]'
