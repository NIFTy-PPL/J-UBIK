import astropy.units as u
from nifty.cl.logger import logger

from .....color import Color
from ...constants import RESOLVE_SPECTRAL_UNIT
from ...parse.data.data_modify import ObservationModify
from ..observation import Observation
from .flagging import flag_weights
from .frequency_handling import (
    freq_average_by_fdom_and_n_freq_chunks,
    restrict_by_freq,
    reverse_frequencies,
)
from .polarization_modify import average_stokesi, restrict_to_stokesi
from .precision import to_double_precision, to_single_precision
from .select_random_visibility_subset import select_random_visibility_subset
from .time_modify import time_average_to_length_of_timebins
from .weight_modify import systematic_error_budget


def modify_observation(
    sky_frequencies: Color, obs: Observation, modify: ObservationModify
) -> Observation:
    """Returns an observation according to ObservationModify. Furthermore,
    if the frequencies are not ordered from smallest to biggest the frequencies
    get reversed. Additionally the visibilities get converted to double
    precission.

    Parameters
    ----------
    sky_frequencies: list[float]
        The frequencies of the sky model, in [Hz].
    obs: Observation
        The observation to be modified
    modify: ObservationModify
        The model for the modification, see `ObservationModify`.
    """

    # Masking
    obs = flag_weights(obs, modify.flag_weights)

    # Reverse the frequencies if they are ordered from high to low.
    if len(obs.freq) > 1:
        if obs.freq[1] - obs.freq[0] < 0:
            obs = reverse_frequencies(obs)

    obs = select_random_visibility_subset(obs, modify.testing_percentage)

    obs = time_average_to_length_of_timebins(obs, modify.time_bins)

    # TODO: Make the two cases into one and supply a None to the function!
    if modify.spectral_min is not None:
        obs = restrict_by_freq(obs, modify.spectral_min, modify.spectral_max)
    if modify.spectral_restrict_to_sky_frequencies:
        freqs_in_unit = sky_frequencies.to(
            RESOLVE_SPECTRAL_UNIT, equivalencies=u.spectral()
        ).value
        obs = restrict_by_freq(obs, freqs_in_unit[0], freqs_in_unit[-1])

    obs = freq_average_by_fdom_and_n_freq_chunks(
        sky_frequencies, obs, modify.spectral_bins
    )
    obs = systematic_error_budget(obs, modify.weight_modify)

    # TODO: None-fy
    if modify.restrict_to_stokes_I:
        logger.info("Restrict to Stokes I")
        obs = restrict_to_stokesi(obs)

    # TODO: None-fy
    if modify.average_to_stokes_I:
        logger.info("Average to Stokes I")
        obs = average_stokesi(obs)

    # TODO: None-fy
    if modify.to_double_precision:
        obs = to_double_precision(obs)

    return obs
