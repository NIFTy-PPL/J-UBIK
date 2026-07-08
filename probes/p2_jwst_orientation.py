"""p2 — JWST orientation truth: which sky direction each array dim carries.

WHAT THIS PROBES
    The shipped JWST interpolation path (world_coordinates_to_index_grid
    with indexing="ij" + build_linear_rotation_and_shift), fed sky points
    at known offsets from the grid center:

    - a bump written at sky[i0+d, j0] must be read back by the point d
      pixels NORTH of center      -> array dim 0 = +Dec
    - a bump written at sky[i0, j0-d] must be read back by the point d
      pixels EAST of center       -> array dim 1 = -RA (j increases West)

    With imshow(origin="lower") this is the standard North-up/East-left
    astronomical orientation.

GOLDEN
    probes/golden/p2_jwst_interpolation.npy — an asymmetric test sky
    interpolated at an off-center subsampled window through the shipped
    path.  First run writes it; later runs assert byte-stable
    reproduction (square-grid behavior must never change).

RUN
    uv run python probes/p2_jwst_orientation.py
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # deterministic, device-independent

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from jubik.wcs.wcs_astropy import WcsAstropy
from jubik.wcs import subsample_pixel_centers, world_coordinates_to_index_grid
from jubik.instruments.jwst.rotation_and_shift.linear_rotation_and_shift import (
    build_linear_rotation_and_shift,
)

GOLDEN = Path(__file__).parent / "golden" / "p2_jwst_interpolation.npy"


def index_coords(points: SkyCoord, wcs: WcsAstropy) -> np.ndarray:
    return np.array(world_coordinates_to_index_grid([points], wcs, "ij")[0])


def main() -> None:
    center = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)
    n = 9
    wcs = WcsAstropy(center=center, shape=(n, n), fov=(n * u.arcsec, n * u.arcsec))
    c = n // 2
    interp = build_linear_rotation_and_shift(indexing="ij", mode="constant")

    north = center.spherical_offsets_by(0 * u.arcsec, 2 * u.arcsec)
    east = center.spherical_offsets_by(2 * u.arcsec, 0 * u.arcsec)
    pts = SkyCoord(
        ra=np.array([[center.ra.deg, north.ra.deg, east.ra.deg]]) * u.deg,
        dec=np.array([[center.dec.deg, north.dec.deg, east.dec.deg]]) * u.deg,
    )
    idx = index_coords(pts, wcs)

    sky = np.zeros((n, n))
    sky[c + 2, c] = 1.0
    reads = np.asarray(interp(sky, idx)).ravel()  # order: center, north, east
    assert np.allclose(reads, [0, 1, 0]), f"dim0 is not +Dec: {reads}"
    print(f"sky[c+2, c]=1 read at (center, north, east) = {reads}")
    print("  -> array dim 0 = +Dec (North)")

    sky = np.zeros((n, n))
    sky[c, c - 2] = 1.0
    reads = np.asarray(interp(sky, idx)).ravel()
    assert np.allclose(reads, [0, 0, 1]), f"dim1 is not -RA: {reads}"
    print(f"sky[c, c-2]=1 read at (center, north, east) = {reads}")
    print("  -> array dim 1 = -RA (j increases toward West; East is left)")

    # --- golden: asymmetric sky through the shipped path -------------
    rng = np.random.default_rng(42)
    sky = rng.normal(size=(n, n))
    sky[c + 3, c - 1] += 5.0
    window = subsample_pixel_centers((2, 7, 1, 8), wcs, subsample=2)
    out = np.asarray(interp(sky, index_coords(window, wcs)))

    GOLDEN.parent.mkdir(exist_ok=True)
    if not GOLDEN.exists():
        np.save(GOLDEN, out)
        print(f"\ngolden WRITTEN: {GOLDEN.name}  shape={out.shape}")
    else:
        np.testing.assert_array_equal(out, np.load(GOLDEN))
        print(f"\ngolden REPRODUCED byte-identically: {GOLDEN.name}")

    print("\nVERDICT: dim0=+Dec, dim1=-RA — North-up/East-left under "
          "imshow(origin='lower').")


if __name__ == "__main__":
    main()
