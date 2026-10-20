import numpy as np
import jubik.instruments.resolve as rve


def generate_random_obs(
    freqs,
    n_rows,
    uv_range,
    w_range,
    polarization_type,
    weight_range=[1, 10],
    fraction_flagged=0.0,
    ant1=None,
    ant2=None,
    times=None,
    rng=None,
):
    # Preserve np.random.seed behavior for existing callers.
    rng = np.random if rng is None else rng
    n_pol = len(polarization_type)
    vis_shape = (n_pol, n_rows, len(freqs))
    vis = rng.normal(size=vis_shape) + 1j * rng.normal(size=vis_shape)
    weight = rng.uniform(weight_range[0], weight_range[1], size=vis_shape)
    u = rng.uniform(*uv_range, size=n_rows)
    v = rng.uniform(*uv_range, size=n_rows)
    w = rng.uniform(*w_range, size=n_rows)
    uvw = np.stack([u, v, w]).T
    flagged = np.where(
        rng.uniform(0, 1, size=vis_shape) < fraction_flagged, True, False
    )
    weight[flagged] = 0
    pol = polarization_type.get_legacy_polarization()

    ant_pos = rve.AntennaPositions(uvw, ant1, ant2, times)
    return rve.Observation(ant_pos, vis, weight, pol, freqs, None)
