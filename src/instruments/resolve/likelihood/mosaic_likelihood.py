from ....parse.instruments.resolve.response import (
    SkyDomain,
    Ducc0Settings,
    FinufftSettings,
    sky_domain_from_grid,
)

from ....grid import Grid
from ..re.response import InterferometryResponse
from ..data.observation import Observation
from ..mosaicing.sky_beamer import SkyBeamerJft

import nifty8.re as jft

from typing import Union


def build_likelihood_from_sky_beamer(
    observation: Observation,
    field_name: str,
    sky_beamer: SkyBeamerJft,
    sky_domain: SkyDomain,
    backend_settings: [Ducc0Settings, FinufftSettings],
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

    likelihood = jft.Gaussian(
        observation.vis.val,
        noise_cov_inv=lambda x: x*observation.weight.val
    )
    return likelihood.amend(response, domain=jft.Vector(sky_beamer.target))


def build_mosaic_likelihoods(
    sky_beamer: SkyBeamerJft,
    observations: list[Observation],
    sky_grid: Grid,
    backend_settings: Union[Ducc0Settings, FinufftSettings],
    direction_key: str = 'REFERENCE_DIR',
):
    '''Build the likelihoods which corresponding to the different observations
    corresponding to different pointings.

    Parameters
    ----------
    sky_beamer: SkyBeamerJft
        The SkyBeamer transforms a sky into a dictionary of different fields
        with applied beam patterns. The keys correspond to different pointings.
    observations: list[Observation]
        A list of observations, which are assumed to hold one pointing (field).
    epsilon:
        The accuracy of the sky2vis operation (Nufft).
    direction_key: str
        The key in the observation which corresponds to the pointing direction.
    '''
    likelihoods = []
    for field_name, beam_direction in sky_beamer.beam_directions.items():
        for o in observations:
            if o.direction_from_key(direction_key) == beam_direction.direction:
                sky_domain = sky_domain_from_grid(
                    sky_grid,
                    (beam_direction.center_x, beam_direction.center_y)
                )

                likelihoods.append(build_likelihood_from_sky_beamer(
                    observation=o,
                    field_name=field_name,
                    sky_beamer=sky_beamer,
                    sky_domain=sky_domain,
                    backend_settings=backend_settings,
                ))
    return likelihoods
