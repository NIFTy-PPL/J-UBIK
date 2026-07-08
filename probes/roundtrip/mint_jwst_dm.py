"""mint_jwst_dm.py — one-time mint of the synthetic JWST roundtrip datamodel.

WHAT THIS MINTS
    probes/golden/roundtrip_jwst_cal.fits — a `jwst.datamodels.ImageModel`
    (256x256) whose `data` array carries the orientation glyph (a letter
    "F"; see probes/roundtrip/glyph.py) painted WORLD-ANCHORED through the
    model's OWN gwcs.  For each glyph sample offset (East, North) in arcsec
    the sky position is `center.spherical_offsets_by(E, N)`, mapped to a
    detector pixel by the gwcs `world_to_pixel` (APE-14 (x, y) order), and
    `data[round(y), round(x)] = 1.0`.  Because the glyph is anchored in
    ABSOLUTE sky coordinates through the real WCS, any later transpose or
    flip introduced by the production loader chain (gwcs -> WcsJwstData ->
    subsample_pixel_centers -> world_coordinates_to_index_grid) shows up as
    a non-identity dihedral verdict when p6_jwst_roundtrip.py scatters the
    data back onto the canonical reconstruction grid.

    The gwcs is a standard FITS-TAN pipeline (Shift | Scale | Pix2Sky_TAN |
    RotateNative2Celestial), NIRCam long-wave pixel scale 0.0630 arcsec,
    CDELT1 negative (RA decreases with +x), center J2000
    ra=204.25 deg dec=-29.87 deg (roughly the radio field; the exact value
    is uncritical but FIXED).  Metadata: NIRCAM / F356W / MJy/sr, with
    wcsinfo ra_ref/dec_ref and pointing ra_v1/dec_v1 set to the center.

    The glyph is scaled (GLYPH_SCALE) so the whole F fits inside the field.
    The SAME center and GLYPH_SCALE are re-used by p6_jwst_roundtrip.py.

RE-RUN ONLY IF the observation format (datamodel/gwcs layout) or the glyph
    changes — never silently.  Regenerating rewrites the frozen golden FITS
    that p6 pins against.

    RUN (from the j-ubik repo root, in the j-ubik venv):
        uv run python probes/roundtrip/mint_jwst_dm.py
"""

from pathlib import Path

import numpy as np
from astropy import coordinates as coord, units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from gwcs import coordinate_frames as cf, wcs as gwcs_wcs
from jwst import datamodels

from glyph import sample_points

# --- frozen mint parameters (shared with p6_jwst_roundtrip.py) -------------
RA0_DEG = 204.25
DEC0_DEG = -29.87
NPIX = 256
PIXSCALE_ARCSEC = 0.0630  # NIRCam long-wave (F356W)
GLYPH_SCALE = 0.6         # shrink the 10" glyph to fit the ~16" field
GLYPH_STEP = 0.05         # dense sampling so strokes stay solid on the recon grid

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "roundtrip_jwst_cal.fits"


def field_center() -> SkyCoord:
    """The fixed J2000 field center (also the glyph anchor)."""
    return SkyCoord(ra=RA0_DEG * u.deg, dec=DEC0_DEG * u.deg)


def build_gwcs() -> gwcs_wcs.WCS:
    """A FITS-TAN gwcs: detector (x, y) -> ICRS sky, RA decreasing with +x."""
    cdelt = PIXSCALE_ARCSEC / 3600.0            # deg / pixel
    crpix = NPIX / 2 - 0.5                       # 0-based pixel of the center
    det2sky = (
        (models.Shift(-crpix) & models.Shift(-crpix))
        | (models.Scale(-cdelt) & models.Scale(cdelt))   # CDELT1 < 0 (RA)
        | models.Pix2Sky_TAN()
        | models.RotateNative2Celestial(RA0_DEG, DEC0_DEG, 180.0)
    )
    det2sky.name = "det2sky"
    detector = cf.Frame2D(name="detector", axes_names=("x", "y"),
                          unit=(u.pix, u.pix))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(), name="icrs",
                            unit=(u.deg, u.deg))
    return gwcs_wcs.WCS([(detector, det2sky), (sky, None)])


def paint_glyph(data: np.ndarray, wcs: gwcs_wcs.WCS, center: SkyCoord) -> None:
    """Paint the glyph into `data` world-anchored through `wcs` (in place)."""
    for east, north in sample_points(scale=GLYPH_SCALE, step=GLYPH_STEP):
        world = center.spherical_offsets_by(east * u.arcsec, north * u.arcsec)
        x, y = wcs.world_to_pixel(world)
        data[int(round(float(y))), int(round(float(x)))] = 1.0


def build_datamodel() -> datamodels.ImageModel:
    """Assemble the ImageModel: painted data, err, metadata, gwcs."""
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


def main() -> None:
    GOLDEN.parent.mkdir(exist_ok=True)
    dm = build_datamodel()
    painted = int((dm.data > 0).sum())
    dm.save(str(GOLDEN))
    print(f"minted {GOLDEN}  ({painted} glyph pixels painted)")

    # --- verify: reload and confirm the painted pixels reproduce the glyph
    from jubik.instruments.jwst.data.jwst_data import JwstData

    jd = JwstData(str(GOLDEN))
    center = field_center()
    errs = []
    for east, north in sample_points(scale=GLYPH_SCALE, step=GLYPH_STEP):
        world = center.spherical_offsets_by(east * u.arcsec, north * u.arcsec)
        x, y = jd.wcs.world_to_pixel(world)
        got = jd.wcs.pixel_to_world(round(float(x)), round(float(y)))
        errs.append(world.separation(got).to(u.arcsec).value)
    max_err = max(errs)
    print(f"JwstData loaded: filter={jd.filter} camera={jd.camera} "
          f"unit={jd.meta.unit} shape={jd.shape}")
    print(f"wcs pixel<->world roundtrip max error {max_err:.4f} arcsec "
          f"(subpixel: pixel scale {PIXSCALE_ARCSEC} arcsec)")
    assert max_err < PIXSCALE_ARCSEC, "painted pixels do not reproduce glyph"
    print("VERIFIED: painted pixels reproduce the glyph world positions.")


if __name__ == "__main__":
    main()
