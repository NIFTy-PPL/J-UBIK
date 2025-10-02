import astropy.units as u
import numpy as np
from nifty.cl.logger import logger
from numpy.typing import NDArray

from .....color import Color, get_2d_binbounds
from ...constants import RESOLVE_SPECTRAL_UNIT
from ..observation import Observation


def freq_average_by_bins(obs: Observation, n_freq_chuncks: int | None):
    if n_freq_chuncks is None:
        return obs

    logger.info(f"Frequency averaging observation to {n_freq_chuncks}")

    splitted_freqs = np.array_split(obs.freq, n_freq_chuncks)
    return freq_average_by_fmin_fmax(obs, splitted_freqs)


def freq_average_by_fdom_and_n_freq_chunks(
    sky_frequencies: Color,
    obs: Observation,
    n_freq_chuncks: int | None,
):
    """Create a new observation with frequencies averaged. The frequencies of
    the new observation will be averaged into `n_freq_chuncks` according to the
    ranges of the sky_frequencies.

    Parameters
    ----------
    sky_frequencies: Color
        The frequency bounds of the sky.
    obs: Observation
        The observation to be modified.
    n_freq_chuncks: int
        The number of frequency chuncks, i.e. the number of averaging bins per
        sky frequency.
    """
    if n_freq_chuncks is None:
        return obs

    fmin_fmax_array = get_2d_binbounds(sky_frequencies, RESOLVE_SPECTRAL_UNIT)

    splitted_freqs = []
    n_obs_in_sky = 0
    for ff in fmin_fmax_array:
        ofreq = restrict_by_freq(obs, ff[0], ff[-1], with_index=False).freq

        if ofreq.size == 0:
            continue
        else:
            n_obs_in_sky += 1

        if len(ofreq) > 2 * n_freq_chuncks:
            tmp_splits = np.array_split(ofreq, n_freq_chuncks)
            for tmp in tmp_splits:
                assert tmp[0] != tmp[-1], (
                    "Frequency chunking of data not compatible with sky frequencies."
                )
                splitted_freqs.append(np.array([tmp[0], tmp[-1]]))

        else:
            assert ofreq[0] != ofreq[-1], (
                "Frequency chunking of data not compatible with sky frequencies."
            )
            splitted_freqs.append(np.array([ofreq[0], ofreq[-1]]))

    obs_out = freq_average_by_fmin_fmax(obs, splitted_freqs)

    freq_len = obs_out.freq.shape[0]
    if freq_len % n_obs_in_sky == 0:
        logger.info(
            "Frequency averaging observation to (N_ObsInSky, N_Chunks) = "
            f"({n_obs_in_sky}, {n_freq_chuncks})"
        )
    else:
        logger.info(
            f"Frequency averaging observation to N_ObsInSky = {n_obs_in_sky}:\n"
            f"    {n_obs_in_sky} ObsInSky a {n_freq_chuncks} N_Chunks"
            f" and {freq_len % n_obs_in_sky} extra."
        )

    return obs_out


def freq_average_by_fmin_fmax(
    obs: Observation,
    fmin_fmax_array: list[float],
):
    splitted_obs = []
    for ff in fmin_fmax_array:
        splitted_obs.append(restrict_by_freq(obs, ff[0], ff[-1]))

    obs_avg = []
    for obsi in splitted_obs:
        new_vis = np.mean(obsi.vis.asnumpy(), axis=2, keepdims=True)
        cov = 1 / obsi.weight.asnumpy()
        new_cov = np.sum(cov, axis=2, keepdims=True) / (obsi.vis.shape[2] ** 2)
        new_weight = 1 / new_cov
        new_freq = np.array([np.mean(obsi.freq)])
        new_obs = Observation(
            obsi.antenna_positions,
            new_vis,
            new_weight,
            obsi.legacy_polarization,
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
        new_vis[:, :, ii] = obs.vis.asnumpy()[:, :, 0]
        new_weight[:, :, ii] = obs.weight.asnumpy()[:, :, 0]

    obs_averaged = Observation(
        obs.antenna_positions,
        new_vis,
        new_weight,
        obs.legacy_polarization,
        new_freq,
        obs._auxiliary_tables,
    )
    return obs_averaged


def get_freqs(
    observation: Observation, frequency_list: list[int] | NDArray, copy=False
):
    """Return observation that contains a subset of the present frequencies.
    Only those whose index is given in the frequency_list.

    Parameters
    ----------
    observation : Observation
    frequency_list : list
        List of indices that shall be returned
    copy: bool
        Whether the underlying arrays are copied.
    """
    mask = np.zeros(observation.nfreq, dtype=bool)
    mask[frequency_list] = 1
    return get_freqs_by_slice(observation, mask, copy)


def get_freqs_by_slice(observation: Observation, slc: slice | NDArray, copy=False):
    """Return observation that contains a subset of the frequencies.
    Only those that are specified by slc.

    Parameters
    ----------
    observation : Observation
    slc: slice | NDArray
        The slice to be returned
    copy: bool
        Whether the underlying arrays are copied.
    """
    # TODO:  Do I need to change something in observation._auxiliary_tables?

    vis = observation._vis[..., slc]
    wgt = observation._weight[..., slc]
    freq = observation._freq[slc]
    if copy:
        vis = vis.copy()
        wgt = wgt.copy()
        freq = freq.copy()

    return Observation(
        observation._antpos,
        vis,
        wgt,
        observation.legacy_polarization,
        freq,
        observation._auxiliary_tables,
    )


def restrict_by_freq(
    observation: Observation, fmin: float, fmax: float, with_index=False
) -> Observation | tuple[Observation, slice]:
    """Return observation that contains a subset of the frequencies.
    Only those that are within fmin and fmax.

    Parameters
    ----------
    observation : Observation
    slc: slice | NDArray
        The slice to be returned
    copy: bool
        Whether the underlying arrays are copied.
    """
    assert all(np.diff(observation.freq) > 0), (
        "The frequencies of the observation need to be increasing"
    )

    start, stop = np.searchsorted(observation.freq, [fmin, fmax])
    ind = slice(start, stop)
    res = get_freqs_by_slice(observation, ind)
    if with_index:
        return res, ind
    return res


def reverse_frequencies(obs: Observation) -> Observation:
    """This reverses the frequencies and returns an observation"""
    logger.info("Reverse frequencies")
    return Observation(
        obs.antenna_positions,
        obs.vis.asnumpy()[:, :, ::-1],
        obs.weight.asnumpy()[:, :, ::-1],
        obs.legacy_polarization,
        obs.freq[::-1],
        auxiliary_tables=obs._auxiliary_tables,
    )


def restrict_to_discontinuous_frequencies(
    obs: Observation, sky_frequencies: Color
) -> Observation:
    """Slicing the observation to conform to discontinuous frequencies.

    Parameters
    ----------
    obs: Observation
        The observation to slice.
    sky_frequencies: Color
        The discontinuous frequency ranges for the sky.
    """

    if sky_frequencies.is_continuous:
        raise ValueError("Only use discontinuous frequencies!")

    freq_indices = []
    for freq in sky_frequencies:
        freq = freq.to(u.Unit(RESOLVE_SPECTRAL_UNIT), equivalencies=u.spectral()).value
        _, find = restrict_by_freq(obs, freq[0], freq[-1], True)
        freq_indices.append(find)

    vis = np.concatenate(
        [obs.vis.asnumpy()[:, :, find] for find in freq_indices], axis=2
    )
    weight = np.concatenate(
        [obs.weight.asnumpy()[:, :, find] for find in freq_indices], axis=2
    )
    freq = np.concatenate([obs.freq[find] for find in freq_indices])

    return Observation(
        antenna_positions=obs.antenna_positions,
        vis=vis,
        weight=weight,
        polarization=obs.legacy_polarization,
        freq=freq,
        auxiliary_tables=obs._auxiliary_tables,
    )
