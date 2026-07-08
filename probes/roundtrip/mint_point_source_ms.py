"""Mint the W7 fixture: a CASA-simulated MS carrying ONE off-center point source.

This one-time minting script produces the two frozen inputs that
``probes/w7_likelihood_position.py`` consumes:

    probes/golden/w7_point_obs.npz   -- a jubik Observation (visibilities)
    probes/golden/w7_truth.json      -- the exact truth in world coordinates

The point of the fixture is an EXTERNAL POSITION ANCHOR that no internal
jubik convention can fake.  CASA paints a single 1 Jy point source onto the
sky at a known OFF-CENTER offset (3" East, 5" North of the phase center)
through a standard FITS RA/Dec WCS, simulates real interferometer
visibilities from it, and we extract those visibilities back through jubik.
Because the source is off-center AND East != North, any conjugation, mirror,
rotation, or axis swap anywhere in the jubik radio + likelihood stack lands
a fit at the WRONG world position -- a conjugation point-reflects it through
the phase center; an axis swap transposes (3,5) -> (5,3).

WHAT IS MINTED
    - A 256x256, 0.25"/pixel truth FITS, phase center J2000 13h37m00s
      -29d52m00s, 333 GHz, single Stokes I plane, BUNIT Jy/pixel.  A single
      1 Jy pixel is painted at (3" East, 5" North) of the phase center via
      the FITS WCS -- well inside the ALMA primary beam (~17" FWHM at
      333 GHz).
    - A noiseless ALMA (alma.cycle1.1.cfg) measurement set, single pointing
      at the phase center, obsmode="int", totaltime 600s, under
      probes/roundtrip/_casa_work/.

REQUIREMENTS
    Stage 1 (model) and Stage 3 (extract) run in a normal python env with
    astropy (Stage 1) and BOTH jubik and python-casacore (Stage 3) importable
    -- the j-ubik venv works once python-casacore is installed into it.
    Stage 2 (CASA sim) runs inside the CASA shell (casatasks), which brings
    its own python; it needs neither jubik nor casacore.

HOW TO RUN (three stages, from the j-ubik repo root)

    CASA=/home/jruestig/Installs/CASA/casa-6.6.3-22-py3.8.el8/bin/casa

    uv run python probes/roundtrip/mint_point_source_ms.py stage-model
    "$CASA" --nogui --nologger -c probes/roundtrip/mint_point_source_ms.py stage-casa
    uv run python probes/roundtrip/mint_point_source_ms.py stage-extract

    (stage-model paints the truth FITS; stage-casa simulates the MS;
     stage-extract reads the MS with ms2observations and freezes the goldens
     w7_point_obs.npz + w7_truth.json.)
"""

import json
import os
import shutil
import sys
from pathlib import Path

# --- geometry / observation constants (single source of truth) -------------

HERE = Path(__file__).resolve().parent
WORK = HERE / "_casa_work"
GOLDEN = HERE.parent / "golden"

TRUTH_FITS = WORK / "w7_point_truth.fits"
GOLDEN_OBS = GOLDEN / "w7_point_obs.npz"
GOLDEN_TRUTH = GOLDEN / "w7_truth.json"

PHASE_CENTER = "J2000 13h37m00s -29d52m00s"  # CASA indirection string
RA_HMS, DEC_DMS = "13h37m00s", "-29d52m00s"  # astropy-parseable twin
FREQ_GHZ = 333.0
NPIX = 256
PIX_ARCSEC = 0.25
FLUX_JY = 1.0

# The known OFF-CENTER truth position of the point source, in world offsets
# from the phase center.  East != North on purpose so a transpose is visible.
SOURCE_EAST_ARCSEC = 3.0
SOURCE_NORTH_ARCSEC = 5.0

# CASA simulation
PROJECT = "w7_point"
ANTENNALIST = "alma.cycle1.1.cfg"
CONFIG_NAME = "alma.cycle1.1"  # antennalist stem -> MS name component
TOTALTIME = "600s"
INTEGRATION = "10s"


# === Stage 1: paint the truth FITS =========================================


def stage_model() -> None:
    """Paint a single 1 Jy point source through a standard RA/Dec WCS."""
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

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
    world = center.spherical_offsets_by(
        SOURCE_EAST_ARCSEC * u.arcsec, SOURCE_NORTH_ARCSEC * u.arcsec
    )
    x, y = wcs.world_to_pixel(world)  # 0-based (col, row)
    xi, yi = int(round(float(x))), int(round(float(y)))
    if not (0 <= yi < NPIX and 0 <= xi < NPIX):
        raise RuntimeError(f"source pixel ({xi},{yi}) fell off the grid -- WCS bug")
    data[yi, xi] = FLUX_JY

    fits.PrimaryHDU(data=data, header=hdr).writeto(TRUTH_FITS, overwrite=True)
    print(f"stage-model: wrote {TRUTH_FITS}")
    print(f"  source at FITS pixel (col={xi}, row={yi}), "
          f"total flux {data.sum():.4f} Jy")
    print(f"  truth offset: East {SOURCE_EAST_ARCSEC}\", "
          f"North {SOURCE_NORTH_ARCSEC}\"")


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

    truth = {
        "phase_center": PHASE_CENTER,
        "phase_center_ra_hms": RA_HMS,
        "phase_center_dec_dms": DEC_DMS,
        "east_arcsec": SOURCE_EAST_ARCSEC,
        "north_arcsec": SOURCE_NORTH_ARCSEC,
        "flux_jy": FLUX_JY,
        "freq_ghz": FREQ_GHZ,
        "mint_npix": NPIX,
        "mint_pix_arcsec": PIX_ARCSEC,
    }
    with open(GOLDEN_TRUTH, "w") as f:
        json.dump(truth, f, indent=2)

    print(f"stage-extract: wrote {GOLDEN_OBS}")
    print(f"stage-extract: wrote {GOLDEN_TRUTH}")
    print(f"  truth: East {SOURCE_EAST_ARCSEC}\", North "
          f"{SOURCE_NORTH_ARCSEC}\", flux {FLUX_JY} Jy")


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


if __name__ == "__main__":
    main()
