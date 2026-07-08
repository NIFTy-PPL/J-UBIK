"""p3 — radio orientation truth: what the resolve gridders do with the array.

WHAT THIS PROBES
    The shipped resolve backends (interferometry_response_ducc with its
    hard-coded flip_v=True, and interferometry_response_finufft), fed a
    unit point source at known array pixels, judged against the standard
    measurement equation

        V(u, v) = integral I(l, m) exp(-2*pi*i*(u*l + v*m)) dl dm,
        l = offset toward East (+RA direction), m = toward North (+Dec),

    which is the physical anchor (same sign convention the resolve
    phase-offset code uses: exp(-2*pi*i*(u*center_x + v*center_y)) with
    center_x = r*sin(PA) = East, center_y = r*cos(PA) = North).

    Stage 1  pins the center pixel (the array index where all phases
             vanish).
    Stage 2  fits the (axis, sign) mapping: which array dim carries l,
             which carries m, and with which signs.  Anisotropic pixel
             sizes (pixsize_x != pixsize_y) make the axis assignment
             observable on a square grid.
    Stage 3  compares against the normative convention (probes/README):
             dim 0 = +Dec = +m,  dim 1 = -RA = -l.

GOLDEN
    probes/golden/p3_radio_vis.npy — ducc visibilities of a fixed random
    sky.  First run writes, later runs assert byte-stable reproduction.

RUN
    uv run python probes/p3_radio_orientation.py
"""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # deterministic, device-independent

import numpy as np

from jubik.instruments.resolve.response import (
    interferometry_response_ducc,
    interferometry_response_finufft,
)

GOLDEN = Path(__file__).parent / "golden" / "p3_radio_vis.npy"

C_LIGHT = 299792458.0
NPIX = 32
PSX, PSY = 1.0e-5, 1.5e-5           # radians; anisotropic on purpose
UVW = np.array([                     # meters; freq=c => u_meters = u_lambda
    (3000.0, 0.0, 0.0),
    (7000.0, 0.0, 0.0),
    (0.0, 3000.0, 0.0),
    (0.0, 7000.0, 0.0),
    (2000.0, 4000.0, 0.0),
    (-4000.0, 2500.0, 0.0),
])
OBS = SimpleNamespace(uvw=UVW, freq=np.array([C_LIGHT]))


def backends() -> dict:
    ducc = interferometry_response_ducc(
        OBS, npix_x=NPIX, npix_y=NPIX, pixsize_x=PSX, pixsize_y=PSY,
        do_wgridding=False, epsilon=1e-9, nthreads=1, verbosity=0,
    )
    finufft = interferometry_response_finufft(
        OBS, pixsize_x=PSX, pixsize_y=PSY, epsilon=1e-9,
        center_x=0.0, center_y=0.0,
    )
    return {"ducc": lambda s: np.asarray(ducc(s)).ravel(),
            "finufft": lambda s: np.asarray(finufft(s)).ravel()}


def point_sky(i: int, j: int) -> np.ndarray:
    sky = np.zeros((NPIX, NPIX))
    sky[i, j] = 1.0
    return sky


def predicted(l: float, m: float) -> np.ndarray:
    u, v = UVW[:, 0], UVW[:, 1]
    return PSX * PSY * np.exp(-2j * np.pi * (u * l + v * m))


def main() -> None:
    R = backends()

    # --- Stage 1: center pixel ----------------------------------------
    centers = {}
    for name, apply in R.items():
        cands = [(NPIX // 2, NPIX // 2), (NPIX // 2 - 1, NPIX // 2 - 1),
                 (NPIX // 2, NPIX // 2 - 1), (NPIX // 2 - 1, NPIX // 2)]
        phases = {c: np.max(np.abs(np.angle(apply(point_sky(*c))))) for c in cands}
        center = min(phases, key=phases.get)
        assert phases[center] < 1e-6, f"{name}: no phase-free center pixel {phases}"
        centers[name] = center
        print(f"1  {name:8s} center pixel = {center}  (max|phase| = {phases[center]:.1e})")

    # --- Stage 2: (axis, sign) mapping --------------------------------
    probes_px = [(6, 0), (0, 4), (5, -3)]   # (di, dj) offsets from center
    hypotheses = [(axes, sl, sm)
                  for axes in ("dim0=l,dim1=m", "dim0=m,dim1=l")
                  for sl in (+1, -1) for sm in (+1, -1)]
    matches = {}
    for name, apply in R.items():
        ci, cj = centers[name]
        vis = [apply(point_sky(ci + di, cj + dj)) for di, dj in probes_px]
        surviving = []
        for axes, sl, sm in hypotheses:
            ok = True
            for (di, dj), v in zip(probes_px, vis):
                if axes == "dim0=l,dim1=m":
                    l, m = sl * di * PSX, sm * dj * PSY
                else:
                    l, m = sl * dj * PSY, sm * di * PSX
                ok = ok and np.allclose(v, predicted(l, m), rtol=1e-4, atol=1e-13)
            if ok:
                surviving.append((axes, sl, sm))
        assert len(surviving) == 1, f"{name}: mapping not unique: {surviving}"
        matches[name] = surviving[0]
        axes, sl, sm = surviving[0]
        print(f"2  {name:8s} {axes}  sign(l)={sl:+d}  sign(m)={sm:+d}")

    assert matches["ducc"] == matches["finufft"], "backends disagree!"

    # --- Stage 3: verdict vs the normative convention ------------------
    axes, sl, sm = matches["ducc"]
    if axes == "dim0=m,dim1=l":
        reading = (f"dim0 = {'+' if sm > 0 else '-'}m ({'North' if sm > 0 else 'South'}), "
                   f"dim1 = {'+' if sl > 0 else '-'}l ({'East' if sl > 0 else 'West'})")
        complies = (sm == +1 and sl == -1)
    else:
        reading = (f"dim0 = {'+' if sl > 0 else '-'}l ({'East' if sl > 0 else 'West'}), "
                   f"dim1 = {'+' if sm > 0 else '-'}m ({'North' if sm > 0 else 'South'})")
        complies = False
    print(f"\n3  radio effective reading: {reading}")
    print(f"   normative (JWST-verified): dim0 = +Dec/North, dim1 = -RA/West")
    print(f"   VERDICT: {'COMPLIES' if complies else 'DIVERGES from the normative convention'}")

    # --- golden --------------------------------------------------------
    rng = np.random.default_rng(7)
    sky = rng.normal(size=(NPIX, NPIX)) ** 2
    out = R["ducc"](sky)
    GOLDEN.parent.mkdir(exist_ok=True)
    if not GOLDEN.exists():
        np.save(GOLDEN, out)
        print(f"\ngolden WRITTEN: {GOLDEN.name}")
    else:
        np.testing.assert_array_equal(out, np.load(GOLDEN))
        print(f"\ngolden REPRODUCED byte-identically: {GOLDEN.name}")


if __name__ == "__main__":
    main()
