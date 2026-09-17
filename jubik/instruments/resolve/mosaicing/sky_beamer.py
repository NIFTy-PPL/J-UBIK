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
from typing import Callable

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from ..constants import RESOLVE_SPECTRAL_UNIT
from ..data.data_modify.frequency import restrict_by_freq
from ..data.direction import Direction
from ..data.observation import Observation
from ..util import calculate_phase_offset_to_image_center
from ....wcs.wcs_astropy import WcsAstropy


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


def build_sky_beamer(
    sky_shape_with_dtype: jft.ShapeWithDtype,
    sky_wcs: WcsAstropy,
    sky_frequency_means: u.Quantity,
    observations: list[Observation],
    beam_func: Callable[float, float],
    direction_key: str = "REFERENCE_DIR",
    field_name_prefix: str = "",
) -> SkyBeamer:
    """Builds the SkyBeamer. The SkyBeamer contains holds an array for each
    pointing containing the beam pattern for the mean of all
    `sky_frequency_means`.

    Beams pair index-for-index with the canonical sky: ``beam[..., i, j]`` is
    the beam at the world position of sky pixel ``[i, j]`` as given by
    ``sky_wcs``.

    Parameters
    ----------
    sky_shape_with_dtype:
        Polarization, Time, Frequency, Sky. The trailing (Sky) axes must equal
        ``sky_wcs.shape_yx``.

    sky_wcs:
        The reconstruction grid's spatial WCS. Provides the pixel grid, the sky
        center and the pixel-to-world mapping.

    sky_frequency_means: u.Quantity
        The binbounds of the reconstruction sky required to be in Hz.

    observations:
        The observations containing the different pointings of the instrument.
        Only the pointings direction is used to set up the beam pattern wrt.
        the corresponding pointing.

    beam_func:
        A function which provides the beam pattern for the instrument.
        The function needs the keywords:
            - freq  (different frequencies)
            - x  (relative distances to the pointing center)

    direction_key:
        The key in the measurement set which specifies the pointing direction.

    field_name_prefix:
        Prefix for the `field_name`, prepended to the target of SkyBeamer.
        This is usefull when more than one instrument is used.

    Returns
    -------
    SkyBeamer
        The SkyBeamer is an operator that transforms an input sky into a
        a dictionary of skies with applied beam pattern. The output holds:
         - keys, which are the names of the different fields (pointings) in the
        list of observations.
         - values hold the sky modulated by the beam pattern for that field
           (pointing).
    """

    _, _, fshape, *sshape = sky_shape_with_dtype.shape
    if tuple(sshape) != sky_wcs.shape_yx:
        raise ValueError(
            f"sky trailing shape {tuple(sshape)} does not match sky_wcs.shape_yx "
            f"{sky_wcs.shape_yx}"
        )
    sky_center = sky_wcs.center
    # sky_coords[i, j] is the world position of sky pixel [i, j]
    sky_coords = sky_wcs.indices_yx_to_world(*np.indices(sky_wcs.shape_yx))

    beam_directions = {}
    for ii, oo in enumerate(_filter_pointings_generator(observations, direction_key)):
        direction = oo.direction_from_key(direction_key)

        o_phase_center = SkyCoord(
            direction.phase_center[0] * u.rad,
            direction.phase_center[1] * u.rad,
            frame=sky_center.frame,
        )
        center_x, center_y = calculate_phase_offset_to_image_center(
            sky_center, o_phase_center
        )

        x = sky_coords.separation(o_phase_center)
        x = x.to(u.rad).value
        beam_pointing = []
        for ff in range(fshape):
            freq_mean = (
                sky_frequency_means[ff]
                .to(RESOLVE_SPECTRAL_UNIT, equivalencies=u.spectral())
                .value
            )
            beam = beam_func(freq=freq_mean, x=x)
            beam_pointing.append(beam)

        beam = jnp.array(beam_pointing)
        beam = jnp.broadcast_to(beam, sky_shape_with_dtype.shape)

        field_name = _create_field_name(ii, oo, beam_directions, field_name_prefix)
        beam_directions[field_name] = BeamPattern(
            center_x=center_x, center_y=center_y, beam=beam, direction=direction
        )

    return SkyBeamer(sky_shape_with_dtype, beam_directions)


def _filter_pointings_generator(observations: list[Observation], direction_key: str):
    """Returns only observations with unique pointings."""
    field_pointings = list()

    for obs in observations:
        direction = obs.direction_from_key(direction_key)
        if direction not in field_pointings:
            field_pointings.append(direction)
            yield obs


def _create_field_name(
    ii: int,
    observation: Observation,
    beam_directions: dict,
    field_name_prefix: str,
) -> str:
    field_name = observation.direction.name
    field_name = f"fld_{ii:04}" if field_name == "" else field_name

    if field_name_prefix != "":
        field_name = f"{field_name_prefix}_{field_name}"

    if field_name in beam_directions:
        field_name = f"{field_name}_{ii}"

    return field_name
