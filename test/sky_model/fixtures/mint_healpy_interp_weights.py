# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""Mint ``healpy_interp_weights.npz``, the reference for
``jubik.sky_model.explicit_sources.healpix_bilinear_weights``.

healpy is not a jubik dependency, so its output is frozen here. Run with any
environment that has healpy (minted with healpy 1.18.1, numpy 2.2.4):

    python test/sky_model/fixtures/mint_healpy_interp_weights.py

For each nside the points are random directions plus the cases that need
care: the poles and polar caps, every ring and points just off it, the
equator-cap transition z = +-2/3, and longitudes at and around the wrap.
"""

from pathlib import Path

import healpy as hp
import numpy as np

NSIDES = (1, 2, 4, 16)
EPS = 1e-9


def points(nside, rng):
    rings = np.arange(1, 4 * nside)
    ring_theta = np.arccos(hp.ringinfo(nside, rings)[2])
    theta = np.concatenate([
        np.arccos(rng.uniform(-1, 1, 200)),
        [0.0, np.pi, EPS, np.pi - EPS],
        rng.uniform(0, ring_theta[0], 10),
        rng.uniform(ring_theta[-1], np.pi, 10),
        ring_theta, ring_theta + EPS, ring_theta - EPS,
        np.arccos([2 / 3, -2 / 3]) + EPS, np.arccos([2 / 3, -2 / 3]) - EPS,
    ])
    phi = rng.uniform(-np.pi, 2 * np.pi, theta.size)
    phi[:12] = [0.0, -EPS, 2 * np.pi - EPS, np.pi, -np.pi + EPS, EPS] * 2
    return theta, phi


def main():
    rng = np.random.default_rng(42)
    out = {}
    for nside in NSIDES:
        theta, phi = points(nside, rng)
        pix, wgt = hp.get_interp_weights(nside, theta, phi)
        out |= {f"theta_{nside}": theta, f"phi_{nside}": phi,
                f"pix_{nside}": pix.T, f"wgt_{nside}": wgt.T}
    np.savez_compressed(Path(__file__).with_name("healpy_interp_weights.npz"), **out)


if __name__ == "__main__":
    main()
