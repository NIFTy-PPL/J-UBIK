import numpy as np

from ...data.antenna_positions import AntennaPositions
from ..observation import Observation, tmin_tmax
from nifty.cl.logger import logger


def restrict_by_time(
    observation: Observation, tmin: float, tmax: float, with_index=False
) -> Observation | tuple[Observation, slice]:
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
    # FIXME check that timebins do not overlap
    # time, ant1, ant2
    ts = observation._antpos.time
    row_to_bin_map = np.empty(ts.shape)
    row_to_bin_map[:] = np.nan

    for ii, (lo, hi) in enumerate(list_of_timebins):
        ind = np.logical_and(ts >= lo, ts < hi)
        assert np.all(np.isnan(row_to_bin_map[ind]))
        row_to_bin_map[ind] = ii
    assert np.all(np.diff(row_to_bin_map) >= 0)

    assert np.all(~np.isnan(row_to_bin_map))
    row_to_bin_map = row_to_bin_map.astype(int)

    ant1 = observation._antpos.ant1
    ant2 = observation._antpos.ant2
    assert np.max(ant1) < 1000
    assert np.max(ant2) < 1000
    assert np.max(row_to_bin_map) < np.iinfo(np.dtype("int64")).max / 1000000
    atset = np.array(list(set(zip(ant1, ant2, row_to_bin_map))))
    atset = atset[np.lexsort(atset.T)]
    atset = tuple(map(tuple, atset))
    dct = {aa: ii for ii, aa in enumerate(atset)}
    dct_inv = {yy: xx for xx, yy in dct.items()}
    masterindex = np.array(
        [dct[(a1, a2, tt)] for a1, a2, tt in zip(ant1, ant2, row_to_bin_map)]
    )

    vis, wgt = observation.vis.asnumpy(), observation.weight.asnumpy()
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
    new_times = np.empty(len(atset), dtype=observation._antpos.time.dtype)
    new_uvw[()] = new_times[()] = np.nan
    denom = np.bincount(masterindex)
    # Assumption: Uvw value for averaged data is average of uvw values of finely binned data
    for ii in range(3):
        new_uvw[:, ii] = (
            np.bincount(masterindex, weights=observation._antpos.uvw[:, ii]) / denom
        )
    new_times = np.bincount(
        row_to_bin_map, weights=observation._antpos.time
    ) / np.bincount(row_to_bin_map)
    assert np.sum(np.isnan(new_uvw)) == 0
    assert np.sum(np.isnan(new_times)) == 0

    new_times = new_times[np.array([dct_inv[ii][2] for ii in range(len(atset))])]
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
    if len_tbin is None:
        return obs

    logger.info(f"Time average to {len_tbin} time bins.")

    tmin, tmax = tmin_tmax(obs)
    n_tbins = int((tmax - tmin) // len_tbin + 2)
    tbins_endpoints = np.arange(tmin, n_tbins * len_tbin + tmin, len_tbin)
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
