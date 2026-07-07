"""p1 — metadata alignment: WcsAstropy metadata matches the canonical frame.

WHAT THIS PROBES
    Batch D of the canonical-frame program: `WcsAstropy` takes `shape`
    and `fov` in NUMPY/CANONICAL order — shape = (nDec, nRA), fov =
    (fov_dec, fov_ra) — matching the layout of the sky array they
    describe (dim0 = +Dec, dim1 = -RA; probes/README.md).  Consequences
    pinned here, all on a RECTANGULAR grid so nothing cancels:

    A. header pairing: FITS axis 1 (RA, CDELT1 < 0) is sized by
       shape[1]/fov[1]; axis 2 (Dec) by shape[0]/fov[0].
    B. distances are index-matched: distances[k] describes array dim k
       (this is exactly what charm's `space_from_grid` consumes).
    C. response coherence: the ij index coordinates of the full world
       span land inside the array dims — the historical square-grid-only
       cancellation is healed, rectangles are correct.
    D. `extent()` returns the house-recipe imshow 4-tuple
       (left, right, bottom, top) = (-h1, +h1, -h0, +h0).
    E. `WcsAstropy_from_wcs` reads astropy's array_shape in the correct
       (ny, nx) order and reproduces shape/fov on a rectangle (the old
       `nx, ny = wcs.array_shape` swap is fixed).

HISTORY
    Until Batch D this probe pinned the OPPOSITE: metadata paired
    shape[0] with the RA header axis while the response read dim0 as
    Dec — two quirks cancelling on square grids only (see the history
    in probes/check_frames.py).

RUN
    uv run python probes/p1_metadata_vs_response.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # deterministic, device-independent

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from jubik.wcs.wcs_astropy import WcsAstropy, WcsAstropy_from_wcs
from jubik.wcs import world_coordinates_to_index_grid


def main() -> None:
    center = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)

    # rectangle, anisotropic: nDec=6 rows of 1", nRA=10 cols of 1"
    n_dec, n_ra = 6, 10
    wcs = WcsAstropy(center=center, shape=(n_dec, n_ra),
                     fov=(6 * u.arcsec, 10 * u.arcsec))

    # --- A: header pairing --------------------------------------------
    header = wcs.to_header()
    assert header["CRPIX1"] == n_ra / 2 + 0.5, "RA axis not sized by shape[1]"
    assert header["CRPIX2"] == n_dec / 2 + 0.5, "Dec axis not sized by shape[0]"
    assert header["CDELT1"] < 0, "axis 1 must be RA (negative CDELT)"
    assert np.isclose(-header["CDELT1"] * 3600, 10 / n_ra), \
        "CDELT1 not fov[1]/shape[1]"
    assert np.isclose(header["CDELT2"] * 3600, 6 / n_dec), \
        "CDELT2 not fov[0]/shape[0]"
    print("A  header: axis1(RA) <- shape[1]/fov[1], axis2(Dec) <- shape[0]/fov[0]")

    # --- B: distances index-matched ------------------------------------
    d = wcs.distances.to(u.arcsec).value
    assert np.allclose(d, [6 / n_dec, 10 / n_ra]), \
        f"distances not index-matched to array dims: {d}"
    print("B  distances[k] describes array dim k (space_from_grid-safe)")

    # --- C: response coherence on the rectangle -------------------------
    north = center.spherical_offsets_by(0 * u.arcsec, 2 * u.arcsec)
    east = center.spherical_offsets_by(2 * u.arcsec, 0 * u.arcsec)
    pts = SkyCoord(
        ra=np.array([[center.ra.deg, north.ra.deg, east.ra.deg]]) * u.deg,
        dec=np.array([[center.dec.deg, north.dec.deg, east.dec.deg]]) * u.deg,
    )
    idx = world_coordinates_to_index_grid([pts], wcs, indexing="ij")[0]
    c0, c1 = idx[0].ravel(), idx[1].ravel()  # order: center, north, east
    assert np.isclose(c0[1] - c0[0], 2.0), "dim0 coordinate does not track +Dec"
    assert np.isclose(c1[2] - c1[0], -2.0), "dim1 coordinate does not track -RA"

    corner_lo = wcs.pixel_to_world(0.0, 0.0)
    corner_hi = wcs.pixel_to_world(float(n_ra - 1), float(n_dec - 1))
    span = SkyCoord(
        ra=np.array([[corner_lo.ra.deg, corner_hi.ra.deg]]) * u.deg,
        dec=np.array([[corner_lo.dec.deg, corner_hi.dec.deg]]) * u.deg,
    )
    r = np.array(world_coordinates_to_index_grid([span], wcs, "ij")[0]).reshape(2, 2)
    assert np.allclose(sorted(r[0]), [0, n_dec - 1], atol=1e-6), \
        f"dim0 world span {r[0]} does not match array dim0 (size {n_dec})"
    assert np.allclose(sorted(r[1]), [0, n_ra - 1], atol=1e-6), \
        f"dim1 world span {r[1]} does not match array dim1 (size {n_ra})"
    print(f"C  rectangle ({n_dec}, {n_ra}): world span -> dim0 in [0, {n_dec-1}], "
          f"dim1 in [0, {n_ra-1}] — index ranges match the array")

    # --- D: house-recipe extent ----------------------------------------
    ext = wcs.extent(u.arcsec)
    assert np.allclose(ext, (-5.0, 5.0, -3.0, 3.0)), \
        f"extent not (left,right,bottom,top)=(-h1,h1,-h0,h0): {ext}"
    print("D  extent() = (-h1, +h1, -h0, +h0) — imshow house recipe")

    # --- E: WcsAstropy_from_wcs on a rectangle ---------------------------
    plain = WCS(header)
    plain.pixel_shape = (n_ra, n_dec)  # astropy pixel order (nx, ny)
    rebuilt = WcsAstropy_from_wcs(plain)
    assert tuple(rebuilt.shape) == (n_dec, n_ra), \
        f"from_wcs shape {rebuilt.shape} not numpy-ordered (nDec, nRA)"
    fov = u.Quantity(rebuilt.fov).to(u.arcsec).value
    assert np.allclose(fov, [6.0, 10.0], rtol=1e-3), \
        f"from_wcs fov {fov} not (fov_dec, fov_ra)"
    print("E  WcsAstropy_from_wcs: (ny, nx) read correctly, rectangle-safe")

    print("\nVERDICT: WcsAstropy metadata ALIGNED with the canonical frame; "
          "rectangles coherent end to end.")


if __name__ == "__main__":
    main()
