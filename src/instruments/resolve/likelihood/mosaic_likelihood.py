from typing import Callable, Union

import nifty8.re as jft
from astropy import units as u

from ....grid import Grid
from ....parse.instruments.resolve.response import Ducc0Settings, FinufftSettings
from ...jwst.parametric_model.parametric_prior import ProbabilityConfig
from ..data.observation import Observation
from ..mosaicing.sky_beamer import SkyBeamerJft
from ..re.response import InterferometryResponse
from ..re.phase_shift_correction import (
    PhaseShiftCorrection,
    build_phase_shift_correction_from_config,
)


def create_response_operator(
    domain: dict,
    sky2vis: InterferometryResponse,
    field_name: str,
    shift: PhaseShiftCorrection | None = None,
    cast_to_dtype: Callable | None = None,
):
    response = jft.wrap(sky2vis, field_name)
    if cast_to_dtype:
        response = jft.wrap(lambda x: sky2vis(cast_to_dtype(x)), field_name)

    if shift is not None:
        domain = domain | shift.domain
        return jft.Model(lambda x: shift(x) * response(x), domain=domain)

    return jft.Model(response, domain=domain)


def build_likelihood_from_sky_beamer(
    observation: Observation,
    field_name: str,
    sky_beamer: SkyBeamerJft,
    sky_grid: Grid,
    backend_settings: Union[Ducc0Settings, FinufftSettings],
    phase_shift_correction_config: ProbabilityConfig | None,
    cast_to_dtype: Callable | None = None,
):
    """First, builds response operator, which takes the `field_name` from the
    sky_beamer operator and calculates the visibilities corresponding to the
    observation.
    Second, builds the likelihood operator corresponding to this observation.

    Parameters
    ----------
    observation: Observation
        The observation under question
    field_name:
        The name of the field (pointing)
    sky_beamer:

    Result
    ------
    The likelihood which takes the beam-corrected sky corresponding to the
    `observation`, which gets transformed to visibility space and compared to
    the visibilities in the observation.
    """

    sky2vis = InterferometryResponse(
        observation=observation,
        sky_grid=sky_grid,
        backend_settings=backend_settings,
    )
    shift = build_phase_shift_correction_from_config(
        phase_shift_correction_config,
        observation=observation,
        field_name=field_name,
    )

    response = create_response_operator(
        domain=sky_beamer.target,
        sky2vis=sky2vis,
        field_name=field_name,
        shift=shift,
        cast_to_dtype=cast_to_dtype,
    )

    likelihood = jft.Gaussian(
        observation.vis.val, noise_cov_inv=lambda x: x * observation.weight.val
    )

    return likelihood.amend(response, domain=jft.Vector(response.domain))
