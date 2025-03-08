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


from ....parse.instruments.resolve.re.mosacing.beam_pattern import BeamPatternConfig

from typing import Callable
from numpy.typing import ArrayLike

import numpy as np
from astropy.constants import c as SPEEDOFLIGHT


def compute_primary_beam_pattern(D, d, freq, x):
    """Compute the theoretical primary beam pattern.

    Parameters
    ----------
    D = Dish diameter (in m)
    d = blockage diameter (in m)
    freq = frequency of the observation (in 1/s, Hz)
    x = sin(theta) = angle from pointing on sky (theta in rad)
    """
    import scipy.special as sc

    a = freq / SPEEDOFLIGHT.value
    b = d / D
    x = np.pi * a * D * x
    mask = x == 0.0
    x[mask] = 1
    sol = 2 / (x * (1 - b ** 2)) * (sc.jn(1, x) - b * sc.jn(1, x * b))
    sol[mask] = 1
    return sol * sol


def build_primary_beam_pattern_from_beam_pattern_config(
    bpc: BeamPatternConfig
) -> Callable[[float, ArrayLike], ArrayLike]:
    '''Returns a callable that evaluates the beam pattern for a frequency on
    the sky, i.e. bp(freq, sky_position)
    '''
    return lambda freq, x: compute_primary_beam_pattern(
        D=bpc.dish_size,
        d=bpc.dish_blockage_size,
        freq=freq,
        x=x
    )
