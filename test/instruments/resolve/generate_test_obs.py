import numpy as np
import jubik0.instruments.resolve as rve


def generate_random_obs(
    freqs,
    n_rows,
    uv_range,
    w_range,
    polarization_type,
    weight_range=[1, 10],
    fraction_flagged=0.0,
    # time_range=None,
    # n_antennas=None,
):
    n_pol = len(polarization_type)
    vis_shape = (n_pol, n_rows, len(freqs))
    vis = np.random.normal(size=vis_shape) + 1j * np.random.normal(size=vis_shape)
    weight = np.random.uniform(weight_range[0], weight_range[1], size=vis_shape)
    u = np.random.uniform(*uv_range, size=n_rows)
    v = np.random.uniform(*uv_range, size=n_rows)
    w = np.random.uniform(*w_range, size=n_rows)
    uvw = np.stack([u, v, w]).T
    flagged = np.where(
        np.random.uniform(0, 1, size=vis_shape) < fraction_flagged, True, False
    )
    weight[flagged] = 0
    pol = polarization_type.get_legacy_polarization()

    times = None
    ant1 = None
    ant2 = None
    # if not time_range is None:
    #     times = np.random.uniform(*time_range, size=n_rows)

    ant_pos = rve.AntennaPositions(uvw, ant1, ant2, times)
    return rve.Observation(ant_pos, vis, weight, pol, freqs, None)
