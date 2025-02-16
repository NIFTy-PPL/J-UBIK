import astropy.units as u
from .color import Color, ColorRange
from numpy import inf

gamma_ray = ColorRange(start=Color(inf*u.eV), end=Color(124*u.keV))
x_ray = ColorRange(start=Color(124*u.keV), end=Color(124*u.eV))
ultraviolet = ColorRange(start=Color(124*u.eV), end=Color(3*u.eV))
visible = ColorRange(start=Color(3*u.eV), end=Color(1.7*u.eV))
infrared = ColorRange(start=Color(1.7*u.eV), end=Color(1.24*u.meV))
microwave = ColorRange(start=Color(1.24*u.meV), end=Color(1.24*u.yeV))
radio = ColorRange(start=Color(1.24*u.yeV), end=Color(0*u.eV))

radio_range = ColorRange(start=microwave.start, end=radio.end)
optical_range = ColorRange(start=ultraviolet.start, end=infrared.end)
