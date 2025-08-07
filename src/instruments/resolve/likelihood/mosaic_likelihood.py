from dataclasses import dataclass
from typing import Callable, Union

from numpy.typing import NDArray

import nifty.re as jft
from astropy import units as u

from ....grid import Grid
from ....parse.instruments.resolve.response import Ducc0Settings, FinufftSettings
from ...jwst.parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)
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
    """Create the full response operator.

    The response operator consists of the following pipeline
    1. Field extraction
    2. cast_to_dtype (optional)
    3. sky2vis
    4. shift (optional)

    Parameters
    ----------
    domain: dict,
        The domain has to contain the `field_name`.
    sky2vis: InterferometryResponse,
        FFT and Gridding
    field_name: str,
        The name of the field to be extracted from the (beam corrected) sky.
    shift: PhaseShiftCorrection | None = None,
        (Optional) Phase-shift-correction
    cast_to_dtype: Callable | None = None,
        (Optional) Casting the dtpye, for example float64 -> float32.
    """
    response = jft.wrap(sky2vis, field_name)
    if cast_to_dtype:
        Warning("Casting dtype is not tested")
        response = jft.wrap(lambda x: sky2vis(cast_to_dtype(x)), field_name)

    if shift is not None:
        domain = domain | shift.domain
        return jft.Model(lambda x: shift(x) * response(x), domain=domain)

    return jft.Model(response, domain=domain)


@dataclass
class LikelihoodBuilder:
    """Builder for a radio likelihood

    Attributes
    ----------
    visibilities: NDArray
        The visibilities of the observation.
    weight: NDArray
        The weights of the visibilities of the observation.
    response: jft.Model
        The response operator.
    name: str = "lh_{index}"
        (Optional) string for display in the minimization.
        This will only be displayed when having multiple likelihoods.
    """

    visibilities: NDArray
    weight: NDArray
    response: jft.Model

    @property
    def likelihood(self) -> jft.Likelihood:
        likelihood = jft.Gaussian(
            self.visibilities, noise_cov_inv=lambda x: x * self.weight
        )
        return likelihood.amend(self.response, domain=jft.Vector(self.response.domain))


def build_likelihood_from_sky_beamer(
    observation: Observation,
    field_name: str,
    sky_beamer: SkyBeamerJft,
    sky_grid: Grid,
    backend_settings: Union[Ducc0Settings, FinufftSettings],
    phase_shift_correction_config: CoordinatesCorrectionPriorConfig | None,
    cast_to_dtype: Callable | None = None,
) -> LikelihoodBuilder:
    """Create a likelihood builder corresponding to the `field_name`.

    The building consists of two steps:
    1. Build response operator

    2. Build the

    First, builds response operator, which takes the `field_name` from the
    sky_beamer operator and calculates the visibilities corresponding to the
    observation.
    Second, builds the likelihood operator corresponding to this observation.

    Parameters
    ----------
    observation: Observation
        The observation under question
    field_name: str
        The name of the field (pointing) corresponding to the `observation`.
    sky_beamer: SkyBeamerJft
        The operator that applies the beam pattern to the sky.
        This can potentially hold multiple pointings, that are identified by different
        field_name.
    sky_grid: Grid
        Used for building the InterferometryResponse
    backend_settings: Union[Ducc0Settings, FinufftSettings]
        The algorithm for gridding and fft.
    phase_shift_correction_config: CoordinatesCorrectionPriorConfig | None,
        (Optional) config object containg the priors for a shift correction.
    cast_to_dtype: Callable | None = None,
        (Optional) Casting the dtpye, for example float64 -> float32.

    Returns
    ------
    LikelihoodBuilder
        The likelihood corresponding to the `observation` and the `field_name`.
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

    return LikelihoodBuilder(
        visibilities=observation.vis.val.val,
        weight=observation.weight.val.val,
        response=response,
    )
