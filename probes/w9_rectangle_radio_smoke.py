"""w9 — rectangular-sky smoke test for the radio chain (current contract).

WHAT THIS MEASURES
    Rectangles are where frame confusions stop cancelling, so this
    witness runs the CASA point source (w7 fixture) through the real
    interferometry_response on a genuinely RECTANGULAR grid
    (96 px RA x 128 px Dec, 0.5"/px) and asks the data where the source
    is, via a flux-marginalized chi2 scan over all pixels.

    Under the CURRENT contract (Grid shape/fov pairing shape[0]=RA;
    sky authored in the wgridder frame, dim0 tracking -East, dim1
    tracking +North — the empirical reading pinned by W7), the truth
    (3" E, 5" N) must land at pixel (48-6, 64+10) = (42, 74).

    Measured 2026-07-08 on the reverted code: argmin = (42, 74) exactly;
    the point-reflected and transposed-frame hypotheses are rejected at
    ~4.6x chi2.  The radio chain (Grid -> response -> gridder) is
    INTERNALLY rectangle-consistent.  The rectangle breakage of the
    current code lives at the WCS seams instead (world->index
    coordinates transposed against the array — p1 part C measures it
    live); that is change C4's territory in canonical-plan.md.

    After C1 lands, the expected pixel moves to the canonical reading
    (grid (nDec, nRA), source at (c0+10, c1+6) for 5"N/3"E at 0.5"/px)
    — update the EXPECTED block in the same commit as C1.

RUN
    uv run python probes/w9_rectangle_radio_smoke.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from jubik.grid import Grid
from jubik.instruments.resolve.data.observation import Observation
from jubik.instruments.resolve.parse.response import Ducc0Settings
from jubik.instruments.resolve.response import interferometry_response

GOLDEN_DIR = Path(__file__).parent / "golden"

N_RA, N_DEC = 96, 128            # rectangle on purpose
PIX_ARCSEC = 0.5
CENTER = SkyCoord("13h37m00s", "-29d52m00s", frame="icrs")

# Current contract (pre-C1): Grid shape[0]=RA axis; sky dim0 tracks -East,
# dim1 tracks +North (W7's measured reading).  Truth: 3" E, 5" N.
EXPECTED = (N_RA // 2 - 6, N_DEC // 2 + 10)
REFLECTED = (N_RA // 2 + 6, N_DEC // 2 - 10)
TRANSPOSED = (N_RA // 2 + 10, N_DEC // 2 - 6)


def main() -> None:
    obs = Observation.load(str(GOLDEN_DIR / "w7_point_obs.npz"))
    truth = json.loads((GOLDEN_DIR / "w7_truth.json").read_text())
    grid = Grid.from_shape_and_fov(
        (N_RA, N_DEC),
        u.Quantity((N_RA * PIX_ARCSEC * u.arcsec, N_DEC * PIX_ARCSEC * u.arcsec)),
        frequencies=None, sky_center=CENTER,
    )
    R = interferometry_response(
        obs, grid,
        Ducc0Settings(epsilon=1e-5, do_wgridding=False, nthreads=1, verbosity=0),
    )
    d = np.asarray(obs.vis_val)
    w = np.asarray(obs.weight_val)

    def chi2(i: int, j: int) -> float:
        sky = np.zeros((1, 1, 1, N_RA, N_DEC))
        sky[0, 0, 0, i, j] = 1.0
        m = np.asarray(R(sky))
        num = np.abs(np.sum(w * np.conj(m) * d)) ** 2
        den = np.sum(w * np.abs(m) ** 2)
        return float(np.sum(w * np.abs(d) ** 2) - num / den)

    best, best_c = None, np.inf
    for i in range(2, N_RA - 2, 2):
        for j in range(2, N_DEC - 2, 2):
            c = chi2(i, j)
            if c < best_c:
                best, best_c = (i, j), c

    print(f"rectangle {N_RA}x{N_DEC} (RA x Dec), {PIX_ARCSEC}\"/px; "
          f"truth {truth['east_arcsec']}\"E {truth['north_arcsec']}\"N")
    print(f"expected pixel (current contract): {EXPECTED}")
    print(f"chi2-argmin pixel (stride 2)     : {best}   chi2 {best_c:.4e}")
    print(f"chi2 at expected                 : {chi2(*EXPECTED):.4e}")
    print(f"chi2 at point-reflected          : {chi2(*REFLECTED):.4e}")
    print(f"chi2 at transposed-frame guess   : {chi2(*TRANSPOSED):.4e}")

    assert best == EXPECTED, (
        f"rectangle argmin {best} != expected {EXPECTED} — the radio chain "
        f"is no longer internally rectangle-consistent under the declared "
        f"contract; measure before touching anything."
    )
    print("\nVERDICT: radio chain internally rectangle-consistent under the "
          "current contract; the rectangle breakage of the current code is "
          "at the WCS seams (p1 part C), i.e. change C4's territory.")


if __name__ == "__main__":
    main()
