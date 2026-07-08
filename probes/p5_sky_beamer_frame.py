"""p5 — the sky-beamer contract: beams pair index-for-index with canonical skies.

WHAT THIS PROBES
    build_jft_sky_beamer builds one beam array per pointing and SkyBeamerJft
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
    writes, later runs assert byte-stable reproduction.

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

from jubik.instruments.resolve.mosaicing.sky_beamer import build_jft_sky_beamer

GOLDEN = Path(__file__).parent / "golden" / "p5_beam.npy"

N = 33                                   # odd -> exact single center pixel
C = N // 2
FOV = u.Quantity((33.0 * u.arcsec, 66.0 * u.arcsec))   # (Dec, RA): 1"/px, 2"/px
CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)


def pointing_observation(name: str, direction: SkyCoord) -> SimpleNamespace:
    phase_center = (direction.ra.rad, direction.dec.rad)
    d = SimpleNamespace(phase_center=phase_center, name=name)
    return SimpleNamespace(direction_from_key=lambda key, d=d: d, direction=d)


def build_beams(observations: list) -> dict:
    beamer = build_jft_sky_beamer(
        sky_shape_with_dtype=jft.ShapeWithDtype((1, 1, 1, N, N), np.float64),
        sky_fov=FOV,
        sky_center=CENTER,
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
        np.testing.assert_array_equal(b, np.load(GOLDEN))
        print(f"golden REPRODUCED byte-identically: {GOLDEN.name}")

    print("\nVERDICT: sky-beamer beams COMPLY with the canonical frame "
          "(dim0=+Dec, dim1=-RA).")


if __name__ == "__main__":
    main()
