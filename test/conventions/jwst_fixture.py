"""A synthetic JWST ImageModel carrying the F glyph, built at test time.

The glyph is painted into the model's ``data`` array world-anchored through
the model's own gwcs: each glyph sample offset (East, North) in arcsec goes
``center.spherical_offsets_by(E, N)``, then ``wcs.world_to_pixel`` (APE-14
``(x, y)`` order), then ``data[round(y), round(x)] = 1``.  Only ``jwst`` and
``gwcs`` code touches the pixels, so a transpose or flip anywhere in jubik's
loader chain shows up as a non-identity dihedral verdict in
``test_claims_jwst.py``.

The gwcs is a plain FITS-TAN pipeline (Shift | Scale | Pix2Sky_TAN |
RotateNative2Celestial) at the NIRCam long-wave pixel scale, CDELT1 negative
(RA decreases with +x).  Building it takes well under a second, so nothing is
stored on disk.
"""

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from jwst import datamodels

from glyph import sample_points

RA0_DEG = 204.25
DEC0_DEG = -29.87
NPIX = 256
PIXSCALE_ARCSEC = 0.0630  # NIRCam long-wave (F356W)
GLYPH_SCALE = 0.6         # shrink the 10" glyph to fit the ~16" field
GLYPH_STEP = 0.05         # dense sampling so strokes stay solid on the recon grid


def field_center() -> SkyCoord:
    return SkyCoord(ra=RA0_DEG * u.deg, dec=DEC0_DEG * u.deg)


def build_gwcs() -> gwcs_wcs.WCS:
    """Detector (x, y) -> ICRS sky, RA decreasing with +x."""
    cdelt = PIXSCALE_ARCSEC / 3600.0
    crpix = NPIX / 2 - 0.5  # 0-based pixel of the center
    det2sky = (
        (models.Shift(-crpix) & models.Shift(-crpix))
        | (models.Scale(-cdelt) & models.Scale(cdelt))  # CDELT1 < 0 (RA)
        | models.Pix2Sky_TAN()
        | models.RotateNative2Celestial(RA0_DEG, DEC0_DEG, 180.0)
    )
    det2sky.name = "det2sky"
    detector = cf.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )
    sky = cf.CelestialFrame(
        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
    )
    return gwcs_wcs.WCS([(detector, det2sky), (sky, None)])


def paint_glyph(data: np.ndarray, wcs: gwcs_wcs.WCS, center: SkyCoord) -> None:
    for east, north in sample_points(scale=GLYPH_SCALE, step=GLYPH_STEP):
        world = center.spherical_offsets_by(east * u.arcsec, north * u.arcsec)
        x, y = wcs.world_to_pixel(world)
        data[int(round(float(y))), int(round(float(x)))] = 1.0


def build_datamodel() -> datamodels.ImageModel:
    center = field_center()
    wcs = build_gwcs()

    data = np.zeros((NPIX, NPIX), dtype=np.float32)
    paint_glyph(data, wcs, center)

    dm = datamodels.ImageModel(data=data)
    dm.err = np.ones((NPIX, NPIX), dtype=np.float32) * 0.01
    dm.meta.wcs = wcs
    dm.meta.instrument.name = "NIRCAM"
    dm.meta.instrument.filter = "F356W"
    dm.meta.bunit_data = "MJy/sr"
    dm.meta.wcsinfo.ra_ref = RA0_DEG
    dm.meta.wcsinfo.dec_ref = DEC0_DEG
    dm.meta.pointing.ra_v1 = RA0_DEG
    dm.meta.pointing.dec_v1 = DEC0_DEG
    return dm
