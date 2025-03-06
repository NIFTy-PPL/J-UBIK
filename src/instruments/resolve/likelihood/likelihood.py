from astropy.coordinates import SkyCoord
from astropy import units as u
from functools import reduce

from typing import Callable, Union
from numpy.typing import ArrayLike

from .parse.instrument_likelihood import BeamPatternConfig

import nifty8 as ift
import resolve as rve

import nifty8.re as jft
import jubik0.instruments.resolve.re as jrve
# import resolve.re as jrve

try:
    from jubik0.instruments.jwst.grid import Grid
except (ModuleNotFoundError, ImportError):
    from jubik0.grid import Grid


def build_jax_instrument_likelihood(
    beam_pattern_config: BeamPatternConfig,
    data_kw: str,
    sky_shape_with_dtype: jft.ShapeWithDtype,
    sky_grid: Grid,
    observations: list[rve.Observation],
    backend_settings: Union[jrve.FinufftSettings, jrve.Ducc0Settings],
    direction_key: str = 'PHASE_DIR',  # Is this true for all ALMA data?
):
    '''Create a sum of log-likelihoods for all observations supplied.
    The domain of the summed log-likelihood is the sky.

    Parameters
    ----------
    data_kw: str
        the data to read from the config file
    sky_shape_with_dtype:
        the 2d shape of the sky brightness distribution and its dtype
    sky_grid: Grid
        The Grid with underlying world coordinate system.
    observations:
        List of observations for which to build the response.
    direction_key:
        The keyname for the key holding the phase direction information of the
        observations.
    epsilon:
        The accuracy of the Nufft.
    '''

    # This spectral and spatial unit is required for the construction of the
    # sky2vis operator.
    SPECTRAL_UNIT = u.Hz

    beam_func = build_alma_pattern(beam_pattern_config)
    sky_beamer = jrve.build_jft_sky_beamer(
        sky_shape_with_dtype=sky_shape_with_dtype,
        sky_fov=sky_grid.spatial.fov,
        sky_center=sky_grid.spatial.center,
        sky_frequency_binbounds=sky_grid.spectral.binbounds_in(SPECTRAL_UNIT),
        observations=observations,
        beam_func=beam_func,
        direction_key=direction_key,
        field_name_prefix=data_kw,
    )

    likelihoods = jrve.jax_build_mosaic_likelihoods(
        sky_beamer=sky_beamer,
        observations=observations,
        sky_grid=sky_grid,
        backend_settings=backend_settings,
        direction_key=direction_key,
    )

    lh = reduce(lambda x, y: x+y, likelihoods)
    return lh, sky_beamer
