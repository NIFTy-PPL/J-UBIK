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
from ....color import Color

# Source: https://www.eso.org/public/teles-instr/alma/receiver-bands/
# Last updated: 2025-02-16


BAND1 = Color([35, 50] * u.Unit("GHz"))
BAND2 = Color([67, 116] * u.Unit("GHz"))
BAND3 = Color([84, 116] * u.Unit("GHz"))
BAND4 = Color([125, 163] * u.Unit("GHz"))
BAND5 = Color([163, 211] * u.Unit("GHz"))
BAND6 = Color([211, 275] * u.Unit("GHz"))
BAND7 = Color([275, 373] * u.Unit("GHz"))
BAND8 = Color([385, 500] * u.Unit("GHz"))
BAND9 = Color([602, 720] * u.Unit("GHz"))
BAND10 = Color([787, 950] * u.Unit("GHz"))

ALMA_RANGE = Color(u.Quantity([BAND1[0, 0], BAND10[0, -1]]))
