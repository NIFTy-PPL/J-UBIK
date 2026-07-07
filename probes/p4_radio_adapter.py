"""p4 — the radio adapter contract: canonical sky in, physical visibilities out.

WHAT THIS PROBES
    The owner-approved fix (2026-07-06): all skies at the jubik boundary
    are authored in the CANONICAL frame (dim0 = +Dec/North, dim1 =
    -RA/West; see probes/README.md), and the radio response owns ONE
    explicit conversion to whatever the wgridder needs.  This probe pins
    that contract at the adapter seam:

        from jubik.instruments.resolve.response import canonical_sky_to_visibilities

    canonical_sky_to_visibilities(backend_apply, sky_canonical) -> vis
    must produce, for a unit point source di pixels North and dj pixels
    West of center (canonical sky[c+di, c+dj] = 1):

        V(u, v) = d_ra * d_dec * exp(-2*pi*i * (u*l + v*m))
        with  m = +di * d_dec   (North offset)
              l = -dj * d_ra    (dj increases West => negative East offset)

    — the standard measurement equation, same anchor as p3.  A control
    asserts the RAW backends do NOT satisfy this (the adapter is doing
    real work, not a no-op).

    CONSTRAINT pinned alongside: the raw builders
    interferometry_response_ducc / _finufft keep their p3-measured
    behavior byte-identically (p3 golden) — the adapter wraps them, it
    does not change them.

GOLDEN
    probes/golden/p4_adapter_vis.npy — adapter-path visibilities of a
    fixed random canonical sky.  First run writes, later runs assert
    byte-stable reproduction.

RUN
    uv run python probes/p4_radio_adapter.py

STATUS
    Written as the acceptance spec BEFORE the adapter exists; it fails
    with ImportError until the implementation lands.
"""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # deterministic, device-independent

import numpy as np

from jubik.instruments.resolve.response import (
    interferometry_response_ducc,
    interferometry_response_finufft,
    canonical_sky_to_visibilities,
)

GOLDEN = Path(__file__).parent / "golden" / "p4_adapter_vis.npy"

C_LIGHT = 299792458.0
NPIX = 32
D_RA, D_DEC = 1.0e-5, 1.5e-5        # radians; anisotropic on purpose
UVW = np.array([
    (3000.0, 0.0, 0.0),
    (7000.0, 0.0, 0.0),
    (0.0, 3000.0, 0.0),
    (0.0, 7000.0, 0.0),
    (2000.0, 4000.0, 0.0),
    (-4000.0, 2500.0, 0.0),
])
OBS = SimpleNamespace(uvw=UVW, freq=np.array([C_LIGHT]))


def backends() -> dict:
    # identical construction to p3: pixsize_x pairs with the l/RA axis
    ducc = interferometry_response_ducc(
        OBS, npix_x=NPIX, npix_y=NPIX, pixsize_x=D_RA, pixsize_y=D_DEC,
        do_wgridding=False, epsilon=1e-9, nthreads=1, verbosity=0,
    )
    finufft = interferometry_response_finufft(
        OBS, pixsize_x=D_RA, pixsize_y=D_DEC, epsilon=1e-9,
        center_x=0.0, center_y=0.0,
    )
    return {"ducc": lambda s: np.asarray(ducc(s)).ravel(),
            "finufft": lambda s: np.asarray(finufft(s)).ravel()}


def canonical_point_sky(di: int, dj: int) -> np.ndarray:
    c = NPIX // 2
    sky = np.zeros((NPIX, NPIX))
    sky[c + di, c + dj] = 1.0
    return sky


def predicted(di: int, dj: int) -> np.ndarray:
    l, m = -dj * D_RA, +di * D_DEC
    u, v = UVW[:, 0], UVW[:, 1]
    return D_RA * D_DEC * np.exp(-2j * np.pi * (u * l + v * m))


def main() -> None:
    R = backends()
    offsets = [(0, 0), (6, 0), (0, 4), (5, -3)]

    for name, apply in R.items():
        for di, dj in offsets:
            vis = canonical_sky_to_visibilities(apply, canonical_point_sky(di, dj))
            np.testing.assert_allclose(
                vis, predicted(di, dj), rtol=1e-4, atol=1e-13,
                err_msg=f"{name}: adapter contract violated at offset {(di, dj)}",
            )
        print(f"{name:8s} canonical contract holds at offsets {offsets}")

    # control: the raw backend alone must NOT satisfy the contract
    raw = R["ducc"](canonical_point_sky(6, 4))
    assert not np.allclose(raw, predicted(6, 4), rtol=1e-4, atol=1e-13), \
        "raw backend satisfies the canonical contract — adapter is a no-op?"
    print("control: raw backend diverges from the contract (adapter does real work)")

    rng = np.random.default_rng(11)
    sky = rng.normal(size=(NPIX, NPIX)) ** 2
    out = canonical_sky_to_visibilities(R["ducc"], sky)
    GOLDEN.parent.mkdir(exist_ok=True)
    if not GOLDEN.exists():
        np.save(GOLDEN, out)
        print(f"golden WRITTEN: {GOLDEN.name}")
    else:
        np.testing.assert_array_equal(out, np.load(GOLDEN))
        print(f"golden REPRODUCED byte-identically: {GOLDEN.name}")

    print("\nVERDICT: radio adapter COMPLIES with the canonical frame "
          "(dim0=+Dec, dim1=-RA).")


if __name__ == "__main__":
    main()
