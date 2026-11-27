# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %
import jax.numpy as jnp


def charge_migration(charge, kappa):
    """The brighter-fatter effect: charge flows from each pixel to its four edge
    neighbours down the potential of the charge already collected,

        Q'_p = Q_p - sum_n (Q_p + Q_n)/2 (Phi_p - Phi_n),   Phi = kappa Q,

    i.e. ``Q' = Q + kappa/2 L(Q²)`` with ``L`` the four-neighbour Laplacian.
    On the last two axes; no charge crosses the image's edge, so the sum is kept.
    Bright pixels lose charge to their neighbours: a bright star broadens, a
    faint one does not. `kappa` is per unit of `charge` (per DN per integration
    for charge in DN per integration) and broadcasts against it, e.g. one per
    exposure, (n, 1, 1).
    """
    s = 0.5 * kappa * charge**2
    out = charge
    for axis in (-2, -1):
        flow = jnp.diff(s, axis=axis)  # from pixel i + 1 into pixel i
        pad = [(0, 0)] * charge.ndim
        pad[axis] = (0, 1)
        out = out + jnp.pad(flow, pad)
        pad[axis] = (1, 0)
        out = out - jnp.pad(flow, pad)
    return out
