"""Mint the RADIO roundtrip fixture: a CASA-simulated MS carrying the F glyph.

This one-time minting script produces the two frozen inputs that
``probes/p7_radio_roundtrip.py`` consumes:

    probes/golden/roundtrip_radio_obs.npz    -- a jubik Observation
    probes/golden/roundtrip_radio_truth.fits -- the truth sky it was made from

The point of the fixture is a SIGN ANCHOR that no internal jubik convention
can fake: CASA paints the F onto the sky through a standard FITS RA/Dec WCS,
simulates real interferometer visibilities from it, and we extract those
visibilities back through jubik.  Any wrong uvw / visibility-sign / axis-swap
convention anywhere in the jubik radio response then shows up in p7 as a
non-identity dihedral verdict on the dirty image.

WHAT IS MINTED
    - A 256x256, 0.25"/pixel truth FITS, phase center J2000 13h37m00s
      -29d52m00s, 333 GHz, single Stokes I plane, BUNIT Jy/pixel.  The F
      glyph (probes/roundtrip/glyph.py) is painted world-anchored through
      the FITS WCS via glyph.sample_points, its bounding box centered on the
      phase center so the whole letter sits well inside the ALMA primary beam
      (~17" FWHM at 333 GHz).  Total flux ~1 Jy.
    - A noiseless ALMA (alma.cycle1.1.cfg) measurement set, single pointing,
      obsmode="int", totaltime 1200s, under probes/roundtrip/_casa_work/.

REQUIREMENTS
    Stage 1 (model) and Stage 3 (extract) run in a normal python env with
    astropy (Stage 1) and BOTH jubik and python-casacore (Stage 3) importable
    -- the j-ubik venv works once python-casacore is installed into it.
    Stage 2 (CASA sim) runs inside the CASA shell (casatasks), which brings
    its own python; it needs neither jubik nor casacore.

HOW TO RUN (three stages, from the j-ubik repo root)

    # Stage 1 + 3 use the j-ubik venv; Stage 2 uses the CASA shell.
    CASA=/home/jruestig/Installs/CASA/casa-6.6.3-22-py3.8.el8/bin/casa

    uv run python probes/roundtrip/mint_radio_ms.py stage-model
    "$CASA" --nogui --nologger -c probes/roundtrip/mint_radio_ms.py stage-casa
    uv run python probes/roundtrip/mint_radio_ms.py stage-extract

    (stage-model paints the truth FITS; stage-casa simulates the MS;
     stage-extract reads the MS with ms2observations and freezes the goldens.)
"""

import os
import shutil
import sys
from pathlib import Path

# --- geometry / observation constants (single source of truth) -------------

HERE = Path(__file__).resolve().parent
WORK = HERE / "_casa_work"
GOLDEN = HERE.parent / "golden"

TRUTH_FITS = WORK / "roundtrip_truth.fits"
GOLDEN_TRUTH = GOLDEN / "roundtrip_radio_truth.fits"
GOLDEN_OBS = GOLDEN / "roundtrip_radio_obs.npz"

PHASE_CENTER = "J2000 13h37m00s -29d52m00s"  # CASA indirection string
RA_HMS, DEC_DMS = "13h37m00s", "-29d52m00s"  # astropy-parseable twin
FREQ_GHZ = 333.0
NPIX = 256
PIX_ARCSEC = 0.25
TOTAL_FLUX_JY = 1.0

# CASA simulation
PROJECT = "roundtrip"
ANTENNALIST = "alma.cycle1.1.cfg"
CONFIG_NAME = "alma.cycle1.1"  # antennalist stem -> MS name component
TOTALTIME = "1200s"
INTEGRATION = "10s"

# The glyph anchor is placed so the F's bounding box (East [-6, 0],
# North [0, 10] arcsec) is centered on the phase center: anchor offset
# (+3, -5) puts every stroke within ~6" of center -> well inside the beam.
GLYPH_ANCHOR_EAST = 3.0
GLYPH_ANCHOR_NORTH = -5.0
GLYPH_SCALE = 1.0


# === Stage 1: paint the truth FITS =========================================


def stage_model() -> None:
    """Paint the F glyph onto a truth FITS through a standard RA/Dec WCS."""
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    from glyph import sample_points  # local import: probes/roundtrip on sys.path

    WORK.mkdir(parents=True, exist_ok=True)

    center = SkyCoord(RA_HMS, DEC_DMS, frame="icrs")

    # Standard FITS celestial WCS: CDELT1 < 0 (RA decreases with column),
    # CRPIX at the array center, SIN projection (radio-standard).
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

    wcs = WCS(hdr)

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
        raise RuntimeError("glyph painted 0 pixels -- WCS/anchor bug")
    data *= TOTAL_FLUX_JY / n_lit  # normalise to the target total flux

    fits.PrimaryHDU(data=data, header=hdr).writeto(TRUTH_FITS, overwrite=True)
    print(f"stage-model: wrote {TRUTH_FITS}")
    print(f"  {n_lit} lit pixels, total flux {data.sum():.4f} Jy, "
          f"peak {data.max():.4e} Jy/pixel")


# === Stage 2: CASA simulation (runs in the CASA shell) =====================


def stage_casa() -> None:
    """Simulate a noiseless ALMA MS from the truth FITS via simobserve."""
    try:
        simobserve  # noqa: F821  -- provided as a global in the casa shell
        _simobserve = simobserve  # noqa: F821
    except NameError:
        from casatasks import simobserve as _simobserve

    if not TRUTH_FITS.exists():
        raise RuntimeError(
            f"{TRUTH_FITS} missing -- run stage-model first (in the venv)."
        )

    WORK.mkdir(parents=True, exist_ok=True)
    # simobserve writes its project tree under the CWD.
    os.chdir(WORK)
    proj_dir = WORK / PROJECT
    if proj_dir.exists():
        shutil.rmtree(proj_dir)

    _simobserve(
        project=PROJECT,
        skymodel=str(TRUTH_FITS),
        indirection=PHASE_CENTER,
        incell=f"{PIX_ARCSEC}arcsec",
        incenter=f"{FREQ_GHZ}GHz",
        inwidth="50MHz",
        setpointings=True,
        mapsize="0arcsec",          # single pointing at indirection
        obsmode="int",
        antennalist=ANTENNALIST,
        totaltime=TOTALTIME,
        integration=INTEGRATION,
        thermalnoise="",             # noiseless
        graphics="none",
        overwrite=True,
        verbose=True,
    )

    ms = proj_dir / f"{PROJECT}.{CONFIG_NAME}.ms"
    if not ms.exists():
        raise RuntimeError(
            f"expected noiseless MS not found at {ms}; "
            f"project contents: {sorted(p.name for p in proj_dir.iterdir())}"
        )
    print(f"stage-casa: simulated MS at {ms}")


# === Stage 3: extract Observation + freeze goldens =========================


def stage_extract() -> None:
    """Read the MS via ms2observations and freeze the goldens."""
    from jubik.instruments.resolve.data.ms_import import ms2observations

    ms = WORK / PROJECT / f"{PROJECT}.{CONFIG_NAME}.ms"
    if not ms.exists():
        raise RuntimeError(f"{ms} missing -- run stage-casa first (in CASA).")

    obs_list = ms2observations(
        str(ms),
        data_column="DATA",
        with_calib_info=True,
        spectral_window=0,
    )
    observations = [o for o in obs_list if o is not None]
    if len(observations) != 1:
        raise RuntimeError(
            f"expected exactly one field/observation, got {len(observations)}"
        )
    obs = observations[0]
    print(f"stage-extract: Observation vis shape {obs.vis_val.shape}, "
          f"nfreq {obs.nfreq}, npol {obs.npol}")

    GOLDEN.mkdir(parents=True, exist_ok=True)
    obs.save(str(GOLDEN_OBS), True)
    shutil.copyfile(TRUTH_FITS, GOLDEN_TRUTH)
    print(f"stage-extract: wrote {GOLDEN_OBS}")
    print(f"stage-extract: wrote {GOLDEN_TRUTH}")


# === dispatch ==============================================================

_STAGES = {
    "stage-model": stage_model,
    "stage-casa": stage_casa,
    "stage-extract": stage_extract,
}


def main() -> None:
    # Lenient argv scan: the CASA launcher injects its own args around the
    # script name, so pick the stage token out of whatever argv we get.
    requested = [a for a in sys.argv if a in _STAGES]
    if not requested:
        print(__doc__)
        print("ERROR: name a stage: " + ", ".join(_STAGES))
        raise SystemExit(2)
    _STAGES[requested[0]]()


# Make `from glyph import ...` work regardless of CWD.
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    main()
