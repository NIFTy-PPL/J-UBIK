from ..observation import Observation

from nifty8.logger import logger

import numpy as np


def freq_average_by_bins(obs: Observation, n_freq_chuncks: int | None):
    if n_freq_chuncks is None:
        return obs

    logger.info(f"Frequency averaging observation to {n_freq_chuncks}")

    splitted_freqs = np.array_split(obs.freq, n_freq_chuncks)
    return freq_average_by_fmin_fmax(obs, splitted_freqs)


def freq_average_by_fdom_and_n_freq_chunks(
    sky_frequencies: list[float],
    obs: Observation,
    n_freq_chuncks: int,
):
    """Create a new observation with frequencies averaged. The frequencies of
    the new observation will be averaged into `n_freq_chuncks` according to the
    ranges of the sky_frequencies.

    Parameters
    ----------
    sky_frequencies: list[float]
        The frequency bounds of the sky.
    obs: Observation
        The observation to be modified.
    n_freq_chuncks: int
        The number of frequency chuncks.
    """
    if n_freq_chuncks is None:
        return obs

    fmin_fmax_array = [
        (sky_frequencies[ii], sky_frequencies[ii + 1])
        for ii in range(len(sky_frequencies) - 1)
    ]
    splitted_obs_freq = []
    for ff in fmin_fmax_array:
        obs_freq = obs.restrict_by_freq(ff[0], ff[-1]).freq
        if obs_freq.size == 0:
            continue
        splitted_obs_freq.append(obs_freq)

    logger.info(
        "Frequency averaging observation to (N_ObsInSky, N_Chunks) = "
        f"({len(splitted_obs_freq)}, {n_freq_chuncks})"
    )

    splitted_freqs = []
    for ofreq in splitted_obs_freq:
        # TODO : Make it robust against bad sky frequency choices.
        tmp_splits = np.array_split(ofreq, n_freq_chuncks)
        for tmp in tmp_splits:
            assert tmp[0] != tmp[-1], (
                "Frequency chunking not of data not compatible with sky frequencies."
            )
            splitted_freqs.append(np.array([tmp[0], tmp[-1]]))

    return freq_average_by_fmin_fmax(obs, splitted_freqs)


def freq_average_by_fmin_fmax(
    obs: Observation,
    fmin_fmax_array: list[float],
):
    splitted_obs = []
    for ff in fmin_fmax_array:
        splitted_obs.append(obs.restrict_by_freq(ff[0], ff[-1]))

    obs_avg = []
    for obsi in splitted_obs:
        new_vis = np.mean(obsi.vis.val, axis=2, keepdims=True)
        cov = 1 / obsi.weight.val
        new_cov = np.sum(cov, axis=2, keepdims=True) / (obsi.vis.shape[2] ** 2)
        new_weight = 1 / new_cov
        new_freq = np.array([np.mean(obsi.freq)])
        new_obs = Observation(
            obsi.antenna_positions,
            new_vis,
            new_weight,
            obsi.polarization,
            new_freq,
            obs._auxiliary_tables,
        )
        obs_avg.append(new_obs)

    new_freq = [obs.freq[0] for obs in obs_avg]
    new_freq = np.array(new_freq)
    new_vis_shape = (obs.vis.shape[0], obs.vis.shape[1], len(new_freq))
    new_vis = np.zeros(new_vis_shape, obs.vis.dtype)
    new_weight = np.zeros(new_vis_shape, obs.weight.dtype)
    for ii, obs in enumerate(obs_avg):
        new_vis[:, :, ii] = obs.vis.val[:, :, 0]
        new_weight[:, :, ii] = obs.weight.val[:, :, 0]

    obs_averaged = Observation(
        obs.antenna_positions,
        new_vis,
        new_weight,
        obs.polarization,
        new_freq,
        obs._auxiliary_tables,
    )
    return obs_averaged
