"""Phase-center transformations for resolve observations."""

import numpy as np
from astropy import units as u
from astropy.constants import c as speedoflight
from nifty.re import logger

from ...parse.data.data_modify.phase_center import ShiftObservation
from ..observation import Observation


def shift_phase_center(obs: Observation, shift: ShiftObservation | None) -> Observation:
    """Shift the phase center of the visibilities.

    A source which sits at the sky offset `shift = (sx, sy)` relative to the
    center of the sky grid ends up at the center after this operation, i.e. its
    visibilities become constant in phase.

    Note
    ----
    The prefactor is `exp(+2j pi (u * sx - v * sy))`. The relative minus
    between the u and the v term follows the convention of jubik's own imaging
    kernel: a point source at the sky offset `(l, m)` produces the model
    visibilities `exp(-2j pi (u * l - v * m))`, the minus on the v term coming
    from `flip_v=True` in `interferometry_response_ducc`.

    This is a per-visibility phase rotation in the narrow-field limit: the
    `w * (n - 1)` term is neglected and neither uvw nor the auxiliary tables
    (and hence `Observation.direction`) are updated.

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

    prefactor = np.exp(2j * np.pi * (uvw[0] * center_x - uvw[1] * center_y))

    # `prefactor` is always double precision, so the product has to be cast
    # back: `Observation` requires vis and weight to be of the same precision.
    vis = obs.vis.asnumpy()
    vis = (vis * prefactor).astype(vis.dtype, copy=False)

    return Observation(
        antenna_positions=obs.antenna_positions,
        vis=vis,
        weight=obs.weight.asnumpy(),
        polarization=obs.legacy_polarization,
        freq=obs.freq,
        auxiliary_tables=obs._auxiliary_tables,
    )
