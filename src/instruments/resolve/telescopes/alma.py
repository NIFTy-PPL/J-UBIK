# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2025 Max-Planck-Society
# Author: Julian Rüstig


import astropy.units as u
from ....color import ColorRange

# Source: https://www.eso.org/public/teles-instr/alma/receiver-bands/
# Last updated: 2025-02-16


BAND1 = ColorRange(start=35*u.GHz, end=50*u.GHz)
BAND2 = ColorRange(start=67*u.GHz, end=116*u.GHz)
BAND3 = ColorRange(start=84*u.GHz, end=116*u.GHz)
BAND4 = ColorRange(start=125*u.GHz, end=163*u.GHz)
BAND5 = ColorRange(start=163*u.GHz, end=211*u.GHz)
BAND6 = ColorRange(start=211*u.GHz, end=275*u.GHz)
BAND7 = ColorRange(start=275*u.GHz, end=373*u.GHz)
BAND8 = ColorRange(start=385*u.GHz, end=500*u.GHz)
BAND9 = ColorRange(start=602*u.GHz, end=720*u.GHz)
BAND10 = ColorRange(start=787*u.GHz, end=950*u.GHz)

ALMA_RANGE = ColorRange(BAND1.start, BAND10.end)
