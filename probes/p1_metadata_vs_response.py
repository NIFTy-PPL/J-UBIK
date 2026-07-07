"""p1 — metadata vs response: the two convention quirks, isolated.

WHAT THIS PROBES
    Quirk A (metadata):  WcsAstropy.__init__ builds the FITS header with
        CRPIX1/CDELT1 from shape[0]/fov[0] — i.e. the metadata pairs
        spatial axis 0 with the FITS *RA* axis (CDELT1 < 0).
    Quirk B (response):  world_coordinates_to_index_grid(indexing="ij")
        flips astropy's (x, y) pixel coordinates to (y, x), and
        map_coordinates then indexes the sky array's dim 0 with the
        *Dec* pixel coordinate.

    A and B contradict each other; they cancel on square grids and
    produce the standard North-up/East-left layout (pinned in p2).
    On rectangular grids they cannot cancel (index ranges transpose
    against the array — demonstrated at the bottom).

VERDICT PRINTED
    metadata claim:  dim 0 = x/RA        (header pairing)
    response truth:  dim 0 = y/Dec       (ij flip + map_coordinates)
    rectangle (8, 4): interpolation coordinates span transposed ranges.

RUN
    uv run python probes/p1_metadata_vs_response.py
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from jubik.wcs.wcs_astropy import WcsAstropy
from jubik.wcs import world_coordinates_to_index_grid


def main() -> None:
    center = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)

    # --- Quirk A: header pairing -------------------------------------
    s0, s1 = 6, 10
    wcs = WcsAstropy(center=center, shape=(s0, s1),
                     fov=(6 * u.arcsec, 10 * u.arcsec))
    header = wcs.to_header()
    assert header["CRPIX1"] == s0 / 2 + 0.5, "axis 1 no longer sized by shape[0]"
    assert header["CRPIX2"] == s1 / 2 + 0.5
    assert header["CDELT1"] < 0, "axis 1 is the RA axis (negative CDELT)"
    assert header["CDELT2"] > 0
    assert np.isclose(-header["CDELT1"] * 3600, 6 / s0), \
        "CDELT1 built from fov[0]/shape[0]"
    print("A  metadata: header axis 1 (RA, CDELT<0) is sized by shape[0]")
    print("   -> metadata claims: spatial axis 0 = x/RA")

    # --- Quirk B: response indexing ----------------------------------
    sq = WcsAstropy(center=center, shape=(9, 9), fov=(9 * u.arcsec, 9 * u.arcsec))
    north = center.spherical_offsets_by(0 * u.arcsec, 2 * u.arcsec)
    east = center.spherical_offsets_by(2 * u.arcsec, 0 * u.arcsec)
    pts = SkyCoord(
        ra=np.array([[center.ra.deg, north.ra.deg, east.ra.deg]]) * u.deg,
        dec=np.array([[center.dec.deg, north.dec.deg, east.dec.deg]]) * u.deg,
    )
    idx = world_coordinates_to_index_grid([pts], sq, indexing="ij")[0]
    c0, c1 = idx[0].ravel(), idx[1].ravel()  # order: center, north, east

    # channel 0 feeds map_coordinates dim 0; it must move for the NORTH point
    assert np.isclose(c0[1] - c0[0], 2.0), "channel 0 does not track Dec"
    assert np.isclose(c1[1] - c1[0], 0.0)
    assert np.isclose(c1[2] - c1[0], -2.0), "channel 1 does not track -RA"
    assert np.isclose(c0[2] - c0[0], 0.0)
    print("B  response: ij-flipped coords -> channel 0 (= array dim 0) is the")
    print("   Dec pixel coordinate; channel 1 tracks RA (reversed)")
    print("   -> response truth: array dim 0 = y/Dec")

    # --- The contradiction, on a rectangle ---------------------------
    wr = WcsAstropy(center=center, shape=(8, 4), fov=(8 * u.arcsec, 4 * u.arcsec))
    c00 = wr.pixel_to_world(0.0, 0.0)
    c73 = wr.pixel_to_world(7.0, 3.0)
    span = SkyCoord(
        ra=np.array([[c00.ra.deg, c73.ra.deg]]) * u.deg,
        dec=np.array([[c00.dec.deg, c73.dec.deg]]) * u.deg,
    )
    r = np.array(world_coordinates_to_index_grid([span], wr, "ij")[0]).reshape(2, 2)
    print(f"C  rectangle (8, 4): full-extent world span maps to "
          f"dim0 in [0, {r[0].max():.0f}] (array size 8), "
          f"dim1 in [0, {r[1].max():.0f}] (array size 4)")
    assert np.isclose(r[0].max(), 3) and np.isclose(r[1].max(), 7), \
        "rectangle index ranges are no longer transposed — quirks were fixed?"
    print("   -> transposed ranges: silent breakage on non-square grids")

    print("\nVERDICT: metadata (dim0=x/RA) contradicts response (dim0=y/Dec);")
    print("         cancellation is square-grid-only.")


if __name__ == "__main__":
    main()
