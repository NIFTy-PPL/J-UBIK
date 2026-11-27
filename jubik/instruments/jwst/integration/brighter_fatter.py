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
    exposure, (n, 1, 1). First order: valid for ``kappa Q << 1/4``; above
    ``Q = 1 / (4 kappa)`` (2.8e5 DN/int at kappa 9.02e-7) a pixel's charge falls
    as its input grows.
    """
    s = 0.5 * kappa * charge**2
    # an edge pixel is its own outer neighbour: no flow across the edge
    p = jnp.pad(s, [(0, 0)] * (s.ndim - 2) + [(1, 1)] * 2, mode="edge")
    neighbours = (
        p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:]
    )
    return charge + neighbours - 4 * s
