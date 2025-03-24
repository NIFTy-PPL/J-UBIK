from .....parse.instruments.resolve.data.data_modify import ObservationModify

from .restrict_to_testing_percentage import restrict_to_testing_percentage
from .reverse_frequencies import reverse_frequencies
from .time_average import time_average
from .frequency_averaging import freq_average_by_fdom_and_n_freq_chunks
from .weight_modify import weight_modify
from .restrict_and_average_to_stokesi import restrict_and_average_to_stokes_i

from ..observation import Observation


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
        obs = restrict_to_testing_percentage(obs, modify.testing_percentage)

    obs = time_average(obs, modify.time_bins)

    if modify.spectral_min is not None:
        obs = obs.restrict_by_freq(modify.spectral_min, modify.spectral_max)
    if modify.spectral_restrict_to_sky_frequencies:
        obs = obs.restrict_by_freq(sky_frequencies[0], sky_frequencies[-1])

    obs = freq_average_by_fdom_and_n_freq_chunks(
        sky_frequencies, obs, modify.spectral_bins
    )
    obs = weight_modify(obs, modify.weight_modify)

    if modify.restrict_to_stokes_I:
        obs = restrict_and_average_to_stokes_i(obs)

    if modify.to_double_precision:
        obs = obs.to_double_precision()

    return obs
