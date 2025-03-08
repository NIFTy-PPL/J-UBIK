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


from .sky_wcs import build_astropy_wcs

from ..data.observation import Observation
from ..data.direction import Direction

import nifty8.re as jft

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from astropy.coordinates import SkyCoord
from astropy import units as u

from dataclasses import dataclass
from typing import Callable


@dataclass
class BeamPattern:
    center_x: float
    center_y: float
    beam: ArrayLike
    direction: Direction


class SkyBeamerJft(jft.Model):
    '''The SkyBeamer transforms an input sky into a dictionary of skies with
    applied beam pattern.

    The output of the SkyBeamer holds:
     - keys, which are the names of the different fields (pointings) in the
    list of observations.

     - values hold the sky modulated by the beam pattern for that field
       (pointing).
   '''

    def __init__(
        self,
        domain_shape: jft.ShapeWithDtype,
        beam_directions: dict[BeamPattern],
    ):
        self.beam_directions = dict(beam_directions)

        self.target_and_beams = {}
        for target_keys, vv in self.beam_directions.items():
            self.target_and_beams[target_keys] = vv.beam

        super().__init__(domain=domain_shape)

    def __call__(self, x):
        out = {}
        for key, beam in self.target_and_beams.items():
            out[key] = x*beam
        return out

    @classmethod
    def _create_object(cls, domain, beam_directions: dict):
        return cls(domain, beam_directions)

    def __add__(self, other):
        assert self.domain == other.domain
        bd = self.beam_directions | other.beam_directions
        return self._create_object(self.domain, bd)


def build_jft_sky_beamer(
    sky_shape_with_dtype: jft.ShapeWithDtype,
    sky_fov: u.Quantity,
    sky_center: SkyCoord,
    sky_frequency_binbounds: list[float],
    observations: list[Observation],
    beam_func: Callable[float, float],
    direction_key: str = 'REFERENCE_DIR',
    field_name_prefix: str = '',
) -> SkyBeamerJft:
    '''Builds the SkyBeamer. The SkyBeamer contains holds an array for each 
    pointing containing the beam pattern for the mean of all
    `sky_frequency_binbounds`.

    Parameters
    ----------
    sky_shape_with_dtype:
        Polarization, Time, Frequency, Sky

    sky_fov:
        Fov, preferably given in units of [rad]

    sky_center:
        The world coordinate of the Sky reference center.

    sky_frequency_binbounds: list[float],
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
    '''

    _, _, fshape, *sshape = sky_shape_with_dtype.shape

    wcs = build_astropy_wcs(sky_center, sshape, sky_fov)
    sky_coords = wcs.pixel_to_world(
        *np.meshgrid(*[np.arange(s) for s in sshape]))

    beam_directions = {}
    for ii, oo in enumerate(
        _filter_pointings_generator(observations, direction_key)
    ):

        direction = oo.direction_from_key(direction_key)

        o_phase_center = SkyCoord(direction.phase_center[0]*u.rad,
                                  direction.phase_center[1]*u.rad,
                                  frame=sky_center.frame)
        r = sky_center.separation(o_phase_center)
        phi = sky_center.position_angle(o_phase_center)
        center_y = r.to(u.rad).value * np.cos(phi.to(u.rad).value)
        center_x = r.to(u.rad).value * np.sin(phi.to(u.rad).value)

        x = sky_coords.separation(o_phase_center)
        x = x.to(u.rad).value
        beam_pointing = []
        for ff in range(fshape):
            freq_mean = _get_mean_frequency(
                ff, fshape, sky_frequency_binbounds, oo)
            beam = beam_func(freq=freq_mean, x=x)

            # TODO : Why do we need to tranpose?
            # Does this come from the Fourier convention of the radio response?
            beam = np.transpose(beam)
            beam_pointing.append(beam)

        beam = jnp.array(beam_pointing)
        beam = jnp.broadcast_to(beam, sky_shape_with_dtype.shape)

        field_name = _create_field_name(
            ii, oo, beam_directions, field_name_prefix)
        beam_directions[field_name] = BeamPattern(
            center_x=center_x,
            center_y=center_y,
            beam=beam,
            direction=direction
        )

    return SkyBeamerJft(sky_shape_with_dtype, beam_directions)


def _get_mean_frequency(ff, n_freq_bins, f_binbounds, observation):
    '''Get the mean frequency.
        1) If multi-frequency sky: mean of the sky bin
        2) If single-frequency sky: mean of the observation
    '''
    if n_freq_bins == 1:
        o, f_ind = observation.restrict_by_freq(
            f_binbounds[ff], f_binbounds[ff+1], True)
        return o.freq.mean()

    return np.mean((f_binbounds[ff], f_binbounds[ff+1]))


def _filter_pointings_generator(
    observations: list[Observation],
    direction_key: str
):
    '''Returns only observations with unique pointings.'''
    field_pointings = list()

    for obs in observations:
        if obs.direction_from_key(direction_key) not in field_pointings:
            field_pointings.append(obs.direction_from_key(direction_key))
            yield obs


def _create_field_name(
    ii: int,
    observation: Observation,
    beam_directions: dict,
    field_name_prefix: str,
) -> str:
    field_name = observation.direction.name
    field_name = f'fld_{ii:04}' if field_name == '' else field_name

    if field_name_prefix != '':
        field_name = f'{field_name_prefix}_{field_name}'

    if field_name in beam_directions:
        field_name = f"{field_name}_{ii}"

    return field_name
