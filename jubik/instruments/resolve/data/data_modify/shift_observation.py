import numpy as np
from astropy import units as u
from astropy.constants import c as speedoflight
from nifty.re import logger

from ...parse.data.data_modify.shift_observation import ShiftObservation
from ..observation import Observation


def shift_phase_center(obs: Observation, shift: ShiftObservation | None) -> Observation:
    """Shift the visibilities by the shift factor.

    Parameters
    ----------
    obs: Observation
        The observation of which we shift the phase center.
    shift: ShiftObservation
        The shift factor.
    """
    if shift is None:
        return obs

    logger.info(f"Shift phase center by {shift.shift}")
    uvw = np.transpose((obs.uvw[..., None] * obs.freq / speedoflight.value), (0, 2, 1))
    uvw = np.array([uvw[None, :, :, ii] for ii in range(3)])
    assert len(uvw[0].shape) == 3, "Check the polarization axis."

    center_x, center_y = shift.shift.to(u.rad).value

    prefactor = np.exp(-2j * np.pi * (uvw[0] * center_x + uvw[1] * center_y))

    return Observation(
        antenna_positions=obs.antenna_positions,
        vis=obs.vis.asnumpy() * prefactor,
        weight=obs.weight.asnumpy(),
        polarization=obs.legacy_polarization,
        freq=obs.freq,
        auxiliary_tables=obs._auxiliary_tables,
    )
