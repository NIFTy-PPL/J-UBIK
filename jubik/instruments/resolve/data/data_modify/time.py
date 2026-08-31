"""Time-domain transformations for resolve observations."""

import numpy as np

from ...data.antenna_positions import AntennaPositions
from ..observation import Observation, tmin_tmax
from nifty.cl.logger import logger


def restrict_by_time(
    observation: Observation, tmin: float, tmax: float, with_index=False
) -> Observation | tuple[Observation, slice]:
    """Restrict an observation to the half-open time interval `[tmin, tmax)`.

    Rows with `time == tmin` are kept, rows with `time == tmax` are dropped.
    The observation needs to be sorted by time.

    Parameters
    ----------
    observation : Observation
        Time-sorted observation.
    tmin, tmax : float
        Bounds of the time interval. `tmin` is inclusive, `tmax` exclusive.
    with_index : bool
        If True, additionally return the slice of rows that was selected.
    """
    assert all(np.diff(observation.time) >= 0), "Time in observation needs to increase"

    start, stop = np.searchsorted(observation.time, [tmin, tmax])
    ind = slice(start, stop)
    res = observation[ind]
    if with_index:
        return res, ind
    return res


def move_time(observation: Observation, t0: float) -> Observation:
    # FIXME Do I need to change something in observation._auxiliary_tables?
    antpos = observation._antpos.move_time(t0)
    return Observation(
        antpos,
        observation._vis,
        observation._weight,
        observation._polarization,
        observation._freq,
        observation._auxiliary_tables,
    )


def time_average(observation: Observation, list_of_timebins):
    """Average an observation within time bins, separately for every baseline.

    All rows that share a baseline `(ant1, ant2)` and a time bin are combined
    into a single output row. The visibilities are combined with the
    inverse-variance weighted mean, the new weight is the sum of the input
    weights and the new uvw coordinate is the plain mean of the input uvw
    coordinates. The time stamp of an output row is the mean time of all rows
    of its time bin (shared by every baseline of that bin).

    Parameters
    ----------
    observation : Observation
        Time-sorted observation with calibration information.
    list_of_timebins : list of (float, float)
        Non-overlapping half-open time bins `[lo, hi)`, ordered in time. They
        need to cover all time stamps of the observation.
    """
    # time, ant1, ant2
    ts = observation._antpos.time
    row_to_bin_map = np.empty(ts.shape)
    row_to_bin_map[:] = np.nan

    for ii, (lo, hi) in enumerate(list_of_timebins):
        ind = np.logical_and(ts >= lo, ts < hi)
        if not np.all(np.isnan(row_to_bin_map[ind])):
            raise ValueError(
                f"The time bin {ii} ([{lo}, {hi})) does overlap with a "
                "previous time bin. The time bins need to be disjoint."
            )
        row_to_bin_map[ind] = ii

    uncovered = np.isnan(row_to_bin_map)
    if np.any(uncovered):
        raise ValueError(
            f"{np.sum(uncovered)} of {ts.size} rows are not covered by any of "
            f"the given time bins, e.g. the row at the time {ts[uncovered][0]}."
            " The time bins need to cover all time stamps of the observation."
        )
    if not np.all(np.diff(row_to_bin_map) >= 0):
        raise ValueError(
            "The rows of the observation need to be sorted by time and the "
            "time bins need to be ordered in time."
        )
    row_to_bin_map = row_to_bin_map.astype(int)

    ant1 = observation._antpos.ant1
    ant2 = observation._antpos.ant2
    atset = np.array(list(set(zip(ant1, ant2, row_to_bin_map))))
    atset = atset[np.lexsort(atset.T)]
    atset = tuple(map(tuple, atset))
    dct = {aa: ii for ii, aa in enumerate(atset)}
    dct_inv = {yy: xx for xx, yy in dct.items()}
    masterindex = np.array(
        [dct[(a1, a2, tt)] for a1, a2, tt in zip(ant1, ant2, row_to_bin_map)]
    )

    vis, wgt = observation.vis.asnumpy(), observation.weight.asnumpy()
    # Visibilities of flagged data points (weight == 0) are allowed to be
    # non-finite, see `Observation.flags_to_nan`. Since NaN*0 == NaN they must
    # be zeroed before they enter the weighted sums below.
    vis = np.where(wgt > 0.0, vis, 0.0)
    new_vis = np.empty(
        (observation.npol, len(atset), observation.nfreq), dtype=observation.vis.dtype
    )
    new_wgt = np.empty(
        (observation.npol, len(atset), observation.nfreq),
        dtype=observation.weight.dtype,
    )
    for pol in range(observation.npol):
        for freq in range(observation.nfreq):
            enum = np.bincount(
                masterindex, weights=vis[pol, :, freq].real * wgt[pol, :, freq]
            )
            enum = enum + 1j * np.bincount(
                masterindex, weights=vis[pol, :, freq].imag * wgt[pol, :, freq]
            )
            denom = np.bincount(masterindex, weights=wgt[pol, :, freq])
            if np.min(denom) == 0.0:
                raise ValueError("Time bin with total weight 0. detected.")
            new_vis[pol, :, freq] = enum / denom
            new_wgt[pol, :, freq] = denom

    new_uvw = np.empty((len(atset), 3), dtype=observation._antpos.uvw.dtype)
    new_uvw[()] = np.nan
    denom = np.bincount(masterindex)
    # Assumption: Uvw value for averaged data is average of uvw values of finely binned data
    for ii in range(3):
        new_uvw[:, ii] = (
            np.bincount(masterindex, weights=observation._antpos.uvw[:, ii]) / denom
        )
    assert np.sum(np.isnan(new_uvw)) == 0

    # Mean time of all rows of a time bin. Only bins that contain at least one
    # row show up in `bin_of_group`, hence empty time bins are dropped instead
    # of dividing by zero.
    bin_of_group = np.array([dct_inv[ii][2] for ii in range(len(atset))])
    time_sum = np.bincount(row_to_bin_map, weights=observation._antpos.time)
    time_count = np.bincount(row_to_bin_map)
    new_times = time_sum[bin_of_group] / time_count[bin_of_group]
    assert np.sum(np.isnan(new_times)) == 0
    assert np.all(np.diff(new_times) >= 0)

    new_ant1 = np.array([dct_inv[ii][0] for ii in range(len(atset))])
    new_ant2 = np.array([dct_inv[ii][1] for ii in range(len(atset))])
    ap = AntennaPositions(new_uvw, new_ant1, new_ant2, new_times)
    return Observation(
        ap,
        new_vis,
        new_wgt,
        observation._polarization,
        observation._freq,
        observation._auxiliary_tables,
    )


def time_average_to_length_of_timebins(obs: Observation, len_tbin: int | None):
    """Average an observation to time bins of the length `len_tbin`.

    Parameters
    ----------
    obs : Observation
        Time-sorted observation with calibration information.
    len_tbin : int | None
        Length of a single time bin, in the time unit of the observation
        (seconds). This is the length of a bin, not the number of bins. The
        bins start at the first time stamp of the observation and are
        half-open, `[tmin + i*len_tbin, tmin + (i+1)*len_tbin)`, so that the
        last time stamp is always covered. Empty bins are dropped. `None`
        returns the observation unchanged.
    """
    if len_tbin is None:
        return obs

    logger.info(f"Time average to time bins of the length {len_tbin}.")

    tmin, tmax = tmin_tmax(obs)
    n_tbins = int((tmax - tmin) // len_tbin + 2)
    # `np.arange(tmin, n_tbins*len_tbin + tmin, len_tbin)` is not guaranteed to
    # have `n_tbins` entries: its length is computed in floating point, which
    # is inexact for the large time offsets of a measurement set.
    tbins_endpoints = tmin + np.arange(n_tbins) * len_tbin
    unique_times = np.unique(obs.time)
    t_intervals = []
    for ii in range(n_tbins - 1):
        start = tbins_endpoints[ii]
        stop = tbins_endpoints[ii + 1]
        s = start <= unique_times
        b = stop > unique_times
        vis_in_inter = np.any(np.logical_and(s, b))
        if vis_in_inter:
            t_intervals.append([start, stop])

    return time_average(obs, t_intervals)
