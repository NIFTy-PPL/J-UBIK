# from ...mosaicing.sky_wcs import build_astropy_wcs
# from ...mosaicing.sky_beamer import (
#     BeamPattern, _create_field_name, _filter_pointings_generator,
#     _get_mean_frequency)

from ...data.observation import Observation

import nifty8.re as jft
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from typing import Callable


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
    '''Builds the SkyBeamer.

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
