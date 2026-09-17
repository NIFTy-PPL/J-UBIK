"""Analytic anchor shared by the radio seam and claim tests.

For a unit point source ``di`` pixels North and ``dj`` pixels West of the
grid center (canonical ``sky[c + di, c + dj] = 1``) the shipped response
must emit, with ``uvw`` exactly as ``ms2observations`` loads them,

    V(u, v) = d_ra * d_dec * exp(+2 pi i (u l + v m)),
    m = +di * d_dec  (North),   l = -dj * d_ra  (dj increases West).

The ``+2 pi i`` sign is what the CASA witness in ``test_claims_radio.py``
confirms externally; the analytic tests here only check that the code is
consistent with it.  Pixel sizes are anisotropic on purpose so the axis
assignment is observable even on a square grid.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from jubik.grid import Grid
from jubik.instruments.resolve.data import Observation
from jubik.instruments.resolve.data.antenna_positions import AntennaPositions
from jubik.instruments.resolve.parse.response import Ducc0Settings
from jubik.instruments.resolve.response import (
    interferometry_response_ducc,
    interferometry_response_finufft,
)
from jubik.polarization import Polarization

C_LIGHT = 299792458.0
CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)
DUCC = Ducc0Settings(epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=0)

# --- stub backends on a square grid -----------------------------------------
NPIX = 32
# freq = c so u in meters equals u in wavelengths; mixed and negative baselines
UVW = np.array(
    [
        (3000.0, 0.0, 0.0),
        (7000.0, 0.0, 0.0),
        (0.0, 3000.0, 0.0),
        (0.0, 7000.0, 0.0),
        (2000.0, 4000.0, 0.0),
        (-4000.0, 2500.0, 0.0),
        (-3500.0, -1500.0, 0.0),
    ]
)
STUB_OBS = SimpleNamespace(uvw=UVW, freq=np.array([C_LIGHT]))
OFFSETS = [(0, 0), (6, 0), (0, 4), (5, -3), (-4, -2)]
PIXSIZES = [("iso", 1.0e-5, 1.0e-5), ("aniso", 1.0e-5, 1.5e-5)]


def stub_backend(backend, d_ra, d_dec):
    """A raw gridder on the stub observation, returning a flat vis vector."""
    if backend == "ducc":
        op = interferometry_response_ducc(
            STUB_OBS, npix_x=NPIX, npix_y=NPIX, pixsize_x=d_ra, pixsize_y=d_dec,
            do_wgridding=False, epsilon=1e-9, nthreads=1, verbosity=0,
        )
    elif backend == "finufft":
        op = interferometry_response_finufft(
            STUB_OBS, pixsize_x=d_ra, pixsize_y=d_dec, epsilon=1e-9,
            center_x=0.0, center_y=0.0,
        )
    else:
        raise ValueError(backend)
    return lambda s: np.asarray(op(s)).ravel()


def canonical_point_sky(di, dj, shape=(NPIX, NPIX)):
    c0, c1 = shape[0] // 2, shape[1] // 2
    sky = np.zeros(shape)
    sky[c0 + di, c1 + dj] = 1.0
    return sky


def predicted_vis(di, dj, d_ra, d_dec, uvw=UVW):
    l, m = -dj * d_ra, +di * d_dec
    uu, vv = uvw[:, 0], uvw[:, 1]
    return d_ra * d_dec * np.exp(+2j * np.pi * (uu * l + vv * m))


# --- a full Observation + Grid on an anisotropic rectangle -------------------
# nDec = 24 rows of 1", nRA = 32 cols of 1.5"
RECT_SHAPE = (24, 32)
RECT_FOV = (RECT_SHAPE[1] * 1.5 * u.arcsec, RECT_SHAPE[0] * 1.0 * u.arcsec)


def rect_grid():
    return Grid.from_shape_and_fov(
        RECT_SHAPE[::-1], RECT_FOV, frequencies=None, sky_center=CENTER
    )


def make_obs(vis, uvw):
    """Minimal single-channel, Stokes-I, imaging-only Observation."""
    return Observation(
        antenna_positions=AntennaPositions(uvw=np.asarray(uvw, np.float64).copy()),
        vis=np.asarray(vis, np.complex128),
        weight=np.ones(np.shape(vis), np.float64),
        polarization=Polarization.trivial(),
        freq=np.array([C_LIGHT]),
        auxiliary_tables=None,
    )


def point_cube(shape, i, j):
    s = np.zeros((1, 1, 1) + tuple(shape))
    s[0, 0, 0, i, j] = 1.0
    return s


def disk_uvw(seed=7, nrow=3000, umax=8.0e4):
    rng = np.random.default_rng(seed)
    r = umax * np.sqrt(rng.uniform(size=nrow))
    th = rng.uniform(0.0, 2 * np.pi, nrow)
    uvw = np.zeros((nrow, 3))
    uvw[:, 0] = r * np.cos(th)
    uvw[:, 1] = r * np.sin(th)
    return uvw
