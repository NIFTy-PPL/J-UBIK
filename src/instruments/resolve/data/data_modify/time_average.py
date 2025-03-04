from ..observation import Observation, tmin_tmax

from nifty8.logger import logger

import numpy as np


def time_average(obs: Observation, len_tbin: int | None):
    if len_tbin is None:
        return obs

    logger.info(f"Time average to {len_tbin} time bins.")

    tmin, tmax = tmin_tmax(obs)
    n_tbins = int((tmax-tmin) // len_tbin + 2)
    tbins_endpoints = np.arange(tmin, n_tbins*len_tbin+tmin, len_tbin)
    unique_times = np.unique(obs.time)
    t_intervals = []
    for ii in range(n_tbins-1):
        start = tbins_endpoints[ii]
        stop = tbins_endpoints[ii+1]
        s = start <= unique_times
        b = stop > unique_times
        vis_in_inter = np.any(np.logical_and(s, b))
        if vis_in_inter:
            t_intervals.append([start, stop])

    return obs.time_average(t_intervals)
