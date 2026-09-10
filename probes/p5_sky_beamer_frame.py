"""p5 — the sky-beamer contract: beams pair index-for-index with canonical skies.

WHAT THIS PROBES
    build_sky_beamer builds one beam array per pointing and SkyBeamer
    multiplies it elementwise onto the sky before the radio response.  Under
    the canonical frame (probes/README.md: sky dim0 = +Dec/North, dim1 =
    -RA/West) the beam must satisfy

        beam[i, j] = beam_func(separation of canonical sky pixel (i, j)
                               from the pointing center)

    Pinned with a pointing offset 4" North and 4" East of the grid center on
    a 33x33 grid with ANISOTROPIC pixels (1"/px along Dec, 2"/px along RA —
    this makes the fov<->axis pairing observable):

        peak at  i = c + 4  (4" North / 1"per-px  -> +4 rows)
                 j = c - 2  (4" East  / 2"per-px  -> -2 cols; East is
                             the negative-j direction)

    A centered pointing peaks exactly at (c, c) (frame-insensitive control).

    Convention for inputs pinned here alongside: `sky_fov` and the sky's
    trailing shape are NUMPY/CANONICAL-ordered — shape = (nDec, nRA),
    fov = (fov_dec, fov_ra) — matching the sky array they describe.

GOLDEN
    probes/golden/p5_beam.npy — the offset-pointing beam.  First run
    writes, later runs assert reproduction within a tight, absolute
    cross-platform floating-point drift ceiling.

RUN
    uv run python probes/p5_sky_beamer_frame.py

STATUS
    Written as the acceptance spec for Batch B; against pre-Batch-B code
    (the sky_beamer transpose + old axis pairing) the peak lands mirrored
    and this probe FAILS.
"""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # deterministic, device-independent

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import nifty.re as jft

from jubik.instruments.resolve.mosaicing.sky_beamer import build_sky_beamer
from jubik.wcs.wcs_astropy import WcsAstropy

GOLDEN = Path(__file__).parent / "golden" / "p5_beam.npy"

N = 33                                   # odd -> exact single center pixel
C = N // 2
FOV = u.Quantity((66.0 * u.arcsec, 33.0 * u.arcsec))   # public (x, y)
CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)


def pointing_observation(name: str, direction: SkyCoord) -> SimpleNamespace:
    phase_center = (direction.ra.rad, direction.dec.rad)
    d = SimpleNamespace(phase_center=phase_center, name=name)
    return SimpleNamespace(direction_from_key=lambda key, d=d: d, direction=d)


def build_beams(observations: list) -> dict:
    beamer = build_sky_beamer(
        sky_shape_with_dtype=jft.ShapeWithDtype((1, 1, 1, N, N), np.float64),
        sky_wcs=WcsAstropy(center=CENTER, shape=(N, N), fov=FOV),
        sky_frequency_means=u.Quantity([100.0e9] * u.Hz),
        observations=observations,
        beam_func=lambda freq, x: np.exp(-((x / 3.0e-5) ** 2)),
    )
    return {k: np.asarray(v.beam)[0, 0, 0] for k, v in beamer.beam_directions.items()}


def main() -> None:
    centered = pointing_observation("centered", CENTER)
    offset_dir = CENTER.spherical_offsets_by(4.0 * u.arcsec, 4.0 * u.arcsec)  # E, N
    offset = pointing_observation("offset", offset_dir)

    beams = build_beams([centered, offset])

    b = beams["centered"]
    assert b.shape == (N, N)
    peak = np.unravel_index(np.argmax(b), b.shape)
    assert peak == (C, C), f"centered pointing peaks at {peak}, expected {(C, C)}"
    print(f"centered pointing: peak at {peak} == grid center  (control)")

    b = beams["offset"]
    peak = np.unravel_index(np.argmax(b), b.shape)
    expected = (C + 4, C - 2)   # 4" N at 1"/px rows; 4" E at 2"/px cols, East = -j
    assert peak == expected, (
        f"offset pointing (4\" N, 4\" E) peaks at {peak}, expected {expected} "
        "— beam does not pair with the canonical sky frame"
    )
    print(f"offset pointing (4\" N, 4\" E): peak at {peak} == (c+4, c-2)")
    print("  -> beam rows track +Dec at the Dec pixel size, columns track "
        "-RA at the RA pixel size")

    GOLDEN.parent.mkdir(exist_ok=True)
    if not GOLDEN.exists():
        np.save(GOLDEN, b)
        print(f"golden WRITTEN: {GOLDEN.name}")
    else:
        golden = np.load(GOLDEN)
        max_abs_drift = float(np.max(np.abs(b - golden)))
        np.testing.assert_allclose(b, golden, rtol=0.0, atol=3e-12)
        print(f"golden REPRODUCED: {GOLDEN.name} "
              f"(max absolute drift {max_abs_drift:.3g})")

    print("\nVERDICT: sky-beamer beams COMPLY with the canonical frame "
          "(dim0=+Dec, dim1=-RA).")


if __name__ == "__main__":
    main()
