from ....parse.instruments.resolve.response import (
    SkyDomain,
    Ducc0Settings,
    FinufftSettings,
)

from ..re.response import InterferometryResponse
from ..data.observation import Observation
from ..mosaicing.sky_beamer import SkyBeamerJft

import nifty8.re as jft

from typing import Callable, Union


def build_likelihood_from_sky_beamer(
    observation: Observation,
    field_name: str,
    sky_beamer: SkyBeamerJft,
    sky_domain: SkyDomain,
    backend_settings: Union[Ducc0Settings, FinufftSettings],
    cast_to_dtype: Callable | None = None,
):
    '''First, builds response operator, which takes the `field_name` from the
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
    '''

    sky2vis = InterferometryResponse(
        observation=observation,
        sky_domain=sky_domain,
        backend_settings=backend_settings,
    )
    response = jft.wrap(lambda x: sky2vis(x), field_name)
    if cast_to_dtype:
        response = jft.wrap(lambda x: sky2vis(cast_to_dtype(x), field_name))

    likelihood = jft.Gaussian(
        observation.vis.val,
        noise_cov_inv=lambda x: x*observation.weight.val
    )
    return likelihood.amend(response, domain=jft.Vector(sky_beamer.target))
