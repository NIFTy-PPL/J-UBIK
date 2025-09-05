from nifty.cl.logger import logger

from ...parse.data.data_modify import ObservationModify
from ..observation import Observation
from .frequency_handling import (
    freq_average_by_fdom_and_n_freq_chunks,
    reverse_frequencies,
    restrict_by_freq,
)
from .select_random_visibility_subset import select_random_visibility_subset
from .time_modify import time_average_to_length_of_timebins
from .weight_modify import weight_modify
from .precision import to_single_precision, to_double_precision
from .polarization_modify import restrict_to_stokesi, average_stokesi
from .masking import mask_corrupted_weights


def modify_observation(
    sky_frequencies: list[float], obs: Observation, modify: ObservationModify
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

    # Reverse the frequencies if they are ordered from high to low.
    if len(obs.freq) > 1:
        if obs.freq[1] - obs.freq[0] < 0:
            obs = reverse_frequencies(obs)

    if modify.testing_percentage is not None:
        obs = select_random_visibility_subset(obs, modify.testing_percentage)

    obs = time_average_to_length_of_timebins(obs, modify.time_bins)

    if modify.spectral_min is not None:
        obs = restrict_by_freq(obs, modify.spectral_min, modify.spectral_max)
    if modify.spectral_restrict_to_sky_frequencies:
        obs = restrict_by_freq(obs, sky_frequencies[0], sky_frequencies[-1])

    obs = freq_average_by_fdom_and_n_freq_chunks(
        sky_frequencies, obs, modify.spectral_bins
    )
    obs = weight_modify(obs, modify.weight_modify)

    obs = mask_corrupted_weights(obs, modify.mask_corrupted_weights)

    if modify.restrict_to_stokes_I:
        logger.info("Restrict to Stokes I")
        obs = restrict_to_stokesi(obs)

    if modify.average_to_stokes_I:
        logger.info("Average to Stokes I")
        obs = average_stokesi(obs)

    if modify.to_double_precision:
        obs = to_double_precision(obs)

    return obs
