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


from dataclasses import dataclass

import nifty.re as jft
from numpy.typing import ArrayLike

from ..data.direction import Direction


@dataclass
class BeamPattern:
    center_x: float
    center_y: float
    beam: ArrayLike
    direction: Direction


class SkyBeamer(jft.Model):
    """The SkyBeamer transforms an input sky into a dictionary of skies with
    applied beam pattern.

    The output of the SkyBeamer holds:
     - keys, which are the names of the different fields (pointings) in the
    list of observations.

     - values hold the sky modulated by the beam pattern for that field
       (pointing).
    """

    def __init__(
        self,
        domain_shape: jft.ShapeWithDtype,
        beam_directions: dict[BeamPattern],
    ):
        self.beam_directions = dict(beam_directions)
        super().__init__(domain=domain_shape)

    def __call__(self, x):
        return {
            key: x * pattern.beam
            for key, pattern in self.beam_directions.items()
        }

    def __add__(self, other):
        assert self.domain == other.domain
        bd = self.beam_directions | other.beam_directions
        return type(self)(self.domain, bd)


