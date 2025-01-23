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
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
SPEEDOFLIGHT = 299792458.


def str2rad(s):
    """Convert string of number and unit to radian.

    Support the following units: muas mas as amin deg rad.

    Parameters
    ----------
    s : str
        TODO

    """
    c = {
        "muas": AS2RAD * 1e-6,
        "mas": AS2RAD * 1e-3,
        "as": AS2RAD,
        "amin": ARCMIN2RAD,
        "deg": DEG2RAD,
        "rad": 1,
    }
    keys = list(c.keys())
    keys.sort(key=len)
    for kk in reversed(keys):
        nn = -len(kk)
        unit = s[nn:]
        if unit == kk:
            return float(s[:nn]) * c[kk]
    raise RuntimeError("Unit not understood")


def str2val(s):
    """Convert string of number and unit to value.

    Support the following keys: p n mu m (nothing) k M G T

    Parameters
    ----------
    s : str
        TODO

    """
    c = {
        "p": 1e-12,
        "n": 1e-9,
        "mu": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12
    }
    keys = set(c.keys())
    if s[-1] in keys:
        return float(s[:-1]) * c[s[-1]]
    return float(s)
