"""Geometry of the CASA-minted radio witness and the truth sky it was made from.

The radio roundtrip test needs two things: the visibilities CASA simulated
(``fixtures/roundtrip_radio_obs.npz``, minted once by ``mint/mint_radio_ms.py``)
and the truth sky those visibilities were simulated from.  The truth sky is
a deterministic function of the constants below and the glyph, so it is not
stored; ``truth_hdu`` repaints it.  ``mint_radio_ms.py`` calls the same
function, so the fixture and the test cannot drift apart.

This module must import under the CASA shell's python as well (stage-casa of
the mint script), so astropy is imported lazily inside the functions.
"""

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OBS_NPZ = FIXTURES / "roundtrip_radio_obs.npz"

PHASE_CENTER = "J2000 13h37m00s -29d52m00s"  # CASA indirection string
RA_HMS, DEC_DMS = "13h37m00s", "-29d52m00s"  # astropy-parseable twin
FREQ_GHZ = 333.0
NPIX = 256
PIX_ARCSEC = 0.25
TOTAL_FLUX_JY = 1.0

# The glyph anchor is placed so the F's bounding box (East [-6, 0],
# North [0, 10] arcsec) is centered on the phase center: anchor offset
# (+3, -5) puts every stroke within ~6" of center, well inside the ALMA
# primary beam (~17" FWHM at 333 GHz).
GLYPH_ANCHOR_EAST = 3.0
GLYPH_ANCHOR_NORTH = -5.0
GLYPH_SCALE = 1.0

# The CASA run simulated 120 integrations of 10 s.  The stored fixture keeps
# every 15th, that is 8 integrations, 3968 rows.  The dihedral margin and the
# vis-domain correlation are unchanged down to 4 integrations (measured
# 2026-09-16); 8 leaves headroom while keeping the file near 130 kB.
THIN_EVERY = 15

# Reconstruction grid used by the test: 128 * 0.5" = 64" = the truth field.
RECON_NPIX = 128
RECON_PIX_ARCSEC = 0.5


def phase_center():
    from astropy.coordinates import SkyCoord

    return SkyCoord(RA_HMS, DEC_DMS, frame="icrs")


def truth_header():
    """Standard FITS celestial header: CDELT1 < 0, CRPIX at the array center."""
    from astropy.io import fits

    center = phase_center()
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = NPIX  # RA axis  (columns / numpy dim 1)
    hdr["NAXIS2"] = NPIX  # Dec axis (rows    / numpy dim 0)
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CRPIX1"] = NPIX / 2 + 0.5
    hdr["CRPIX2"] = NPIX / 2 + 0.5
    hdr["CRVAL1"] = center.ra.deg
    hdr["CRVAL2"] = center.dec.deg
    hdr["CDELT1"] = -PIX_ARCSEC / 3600.0
    hdr["CDELT2"] = +PIX_ARCSEC / 3600.0
    hdr["RADESYS"] = "ICRS"
    hdr["EQUINOX"] = 2000.0
    hdr["BUNIT"] = "Jy/pixel"
    hdr["RESTFRQ"] = FREQ_GHZ * 1e9
    return hdr


def truth_hdu():
    """The F glyph painted world-anchored through the FITS WCS, 1 Jy total.

    This is exactly the sky CASA simulated from.  Pixel (row, col) order is
    the FITS/numpy one, so the array is already in the canonical frame.
    """
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS

    from glyph import sample_points

    hdr = truth_header()
    wcs = WCS(hdr)
    center = phase_center()

    data = np.zeros((NPIX, NPIX), dtype=np.float32)
    for east, north in sample_points(scale=GLYPH_SCALE):
        d_east = (east + GLYPH_ANCHOR_EAST) * u.arcsec
        d_north = (north + GLYPH_ANCHOR_NORTH) * u.arcsec
        world = center.spherical_offsets_by(d_east, d_north)
        x, y = wcs.world_to_pixel(world)  # 0-based (col, row)
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= yi < NPIX and 0 <= xi < NPIX:
            data[yi, xi] = 1.0

    n_lit = int((data > 0).sum())
    if n_lit == 0:
        raise RuntimeError("glyph painted 0 pixels: WCS or anchor bug")
    data *= TOTAL_FLUX_JY / n_lit
    return fits.PrimaryHDU(data=data, header=hdr)


def thin(observation):
    """Keep every ``THIN_EVERY``-th integration of a full CASA observation."""
    times = np.asarray(observation.time)
    keep = np.isin(times, np.unique(times)[::THIN_EVERY])
    return observation[keep]
