"""p4 — ACCEPTANCE SPEC for change C1: the canonical radio adapter.

STATUS
    This probe FAILS by design (ImportError) until change C1 of
    probes/canonical-plan.md lands.  It is the executable contract that
    C1 must satisfy — written spec-first, before the implementation.

WHAT IT REQUIRES
    All skies at the jubik boundary are authored in the CANONICAL frame
    (dim0 = +Dec/North, dim1 = -RA/West; probes/README.md), and the
    radio response owns ONE explicit conversion to the wgridder layout:

        from jubik.instruments.resolve.response import canonical_sky_to_visibilities

    The conversion is a PURE AXIS TRANSPOSE — no conjugation, no sign
    flips — exact parity with the upstream `resolve` package
    (vol * dirty2vis(sky, flip_v=True); local copy ~/pro/python/resolve)
    fed a canonical sky.  For a unit point source di pixels North and
    dj pixels West of center (canonical sky[c+di, c+dj] = 1) the
    visibilities must satisfy the EFFECTIVE measurement equation of the
    CASA/MS + flip_v pipeline, written with uvw exactly as
    ms2observations loads them:

        V(u, v) = d_ra * d_dec * exp(+2*pi*i * (u*l + v*m))
        with  m = +di * d_dec   (North offset)
              l = -dj * d_ra    (dj increases West => negative East offset)

    Controls pinned alongside:
    - the RAW backends do NOT satisfy the contract (the transpose does
      real work);
    - the CONJUGATED adapter output does NOT satisfy it either — a
      spurious conjugation can never re-enter unnoticed (a conjugated
      forward makes a likelihood fit converge to the rot180 sky; that
      was the 2026-07-06 defect, found by p8/W7-class witnesses).
    - the raw builders interferometry_response_ducc / _finufft keep
      their p3-measured behavior byte-identically (p3 golden) — the
      adapter wraps them, it does not change them.

    CONVENTION NOTE: an earlier version of this contract used the
    textbook exponent exp(-2*pi*i*(ul+vm)) at face value of the loaded
    uvw — the CONJUGATE of what the CASA + flip_v pipeline realizes in
    these variables.  That wrong anchor certified a spurious jnp.conj
    into the (since reverted) adapter.  The sign here is fixed by the
    external witnesses (p7's CASA fixture, the M51 dataset, upstream
    resolve), not by theory written in this repo.

GOLDEN
    probes/golden/p4_adapter_vis.npy — deleted 2026-07-08 (the old file
    froze the wrong-convention output; documented exception to the
    never-regenerate rule).  C1's first passing run re-freezes it;
    later runs assert byte-stable reproduction.

RUN
    uv run python probes/p4_radio_adapter.py
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
    return D_RA * D_DEC * np.exp(+2j * np.pi * (u * l + v * m))


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

    # control 1: the raw backend alone must NOT satisfy the contract
    raw = R["ducc"](canonical_point_sky(6, 4))
    assert not np.allclose(raw, predicted(6, 4), rtol=1e-4, atol=1e-13), \
        "raw backend satisfies the canonical contract — adapter is a no-op?"
    print("control 1: raw backend diverges from the contract "
          "(the transpose does real work)")

    # control 2: the CONJUGATED adapter output must NOT satisfy it either
    vis = canonical_sky_to_visibilities(R["ducc"], canonical_point_sky(6, 4))
    assert not np.allclose(np.conj(vis), predicted(6, 4), rtol=1e-4, atol=1e-13), \
        "conjugated adapter output ALSO satisfies the contract — degenerate?"
    print("control 2: conjugated adapter output violates the contract "
          "(no spurious conjugation present)")

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
          "(dim0=+Dec, dim1=-RA) and the CASA-effective measurement "
          "equation — conjugation-free.")


if __name__ == "__main__":
    main()
