"""Mint the radio roundtrip fixture: a CASA-simulated MS carrying the F glyph.

Produces the one data file committed under ``test/conventions/fixtures``:

    roundtrip_radio_obs.npz    a thinned jubik Observation

The fixture is a sign anchor no jubik convention can fake.  CASA paints the F
onto the sky through a standard FITS RA/Dec WCS, simulates interferometer
visibilities from it, and jubik only reads the result back.  A wrong uvw,
visibility-sign or axis convention anywhere in the radio path then shows up
in ``test_claims_radio.py`` as a non-identity dihedral verdict.

The truth sky is not stored.  ``radio_fixture.truth_hdu`` repaints it from
the same constants, and the test compares against that.

What is simulated: a noiseless ALMA (alma.cycle1.1.cfg) measurement set,
single pointing, obsmode "int", 1200 s in 10 s integrations, 333 GHz, from a
256x256 0.25"/pixel truth image with 1 Jy total flux.  Stage 3 keeps every
``radio_fixture.THIN_EVERY``-th integration before saving.

Re-run only if the glyph, the geometry constants, or the Observation save
format change.  Then update ``CASA_OBS_SHA256`` in ``conftest.py`` in the
same commit (stage 3 prints the new hash).

How to run, three stages, from the repo root:

    CASA=/path/to/casa-6.x/bin/casa

    uv run python test/conventions/mint/mint_radio_ms.py stage-model
    "$CASA" --nogui --nologger -c test/conventions/mint/mint_radio_ms.py stage-casa
    uv run python test/conventions/mint/mint_radio_ms.py stage-extract

Stages 1 and 3 run in the jubik venv with python-casacore installed.  Stage 2
runs inside the CASA shell, which brings its own python; it needs neither
jubik nor casacore, only ``radio_fixture`` (numpy) on the path.
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVENTIONS = HERE.parent
sys.path.insert(0, str(CONVENTIONS))  # radio_fixture, glyph

import radio_fixture as rf  # noqa: E402

WORK = HERE / "_casa_work"
TRUTH_FITS = WORK / "roundtrip_truth.fits"

PROJECT = "roundtrip"
ANTENNALIST = "alma.cycle1.1.cfg"
CONFIG_NAME = "alma.cycle1.1"  # antennalist stem -> MS name component
TOTALTIME = "1200s"
INTEGRATION = "10s"


def stage_model() -> None:
    """Write the truth FITS CASA will simulate from."""
    WORK.mkdir(parents=True, exist_ok=True)
    hdu = rf.truth_hdu()
    hdu.writeto(TRUTH_FITS, overwrite=True)
    data = hdu.data
    print(f"stage-model: wrote {TRUTH_FITS}")
    print(f"  {int((data > 0).sum())} lit pixels, total flux {data.sum():.4f} Jy, "
          f"peak {data.max():.4e} Jy/pixel")


def stage_casa() -> None:
    """Simulate a noiseless ALMA MS from the truth FITS via simobserve."""
    try:
        simobserve  # noqa: F821  (a global in the casa shell)
        _simobserve = simobserve  # noqa: F821
    except NameError:
        from casatasks import simobserve as _simobserve

    if not TRUTH_FITS.exists():
        raise RuntimeError(f"{TRUTH_FITS} missing: run stage-model first.")

    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)  # simobserve writes its project tree under the CWD
    proj_dir = WORK / PROJECT
    if proj_dir.exists():
        shutil.rmtree(proj_dir)

    _simobserve(
        project=PROJECT,
        skymodel=str(TRUTH_FITS),
        indirection=rf.PHASE_CENTER,
        incell=f"{rf.PIX_ARCSEC}arcsec",
        incenter=f"{rf.FREQ_GHZ}GHz",
        inwidth="50MHz",
        setpointings=True,
        mapsize="0arcsec",  # single pointing at indirection
        obsmode="int",
        antennalist=ANTENNALIST,
        totaltime=TOTALTIME,
        integration=INTEGRATION,
        thermalnoise="",  # noiseless
        graphics="none",
        overwrite=True,
        verbose=True,
    )

    ms = proj_dir / f"{PROJECT}.{CONFIG_NAME}.ms"
    if not ms.exists():
        raise RuntimeError(
            f"expected MS not found at {ms}; project contents: "
            f"{sorted(p.name for p in proj_dir.iterdir())}"
        )
    print(f"stage-casa: simulated MS at {ms}")


def stage_extract() -> None:
    """Read the MS via ms2observations, thin, save, print the hash."""
    from jubik.instruments.resolve.data.ms_import import ms2observations

    ms = WORK / PROJECT / f"{PROJECT}.{CONFIG_NAME}.ms"
    if not ms.exists():
        raise RuntimeError(f"{ms} missing: run stage-casa first (in CASA).")

    obs_list = ms2observations(
        str(ms), data_column="DATA", with_calib_info=True, spectral_window=0
    )
    observations = [o for o in obs_list if o is not None]
    if len(observations) != 1:
        raise RuntimeError(
            f"expected exactly one field/observation, got {len(observations)}"
        )
    full = observations[0]
    obs = rf.thin(full)
    print(f"stage-extract: {full.nrow} rows thinned to {obs.nrow} "
          f"(every {rf.THIN_EVERY}th integration), nfreq {obs.nfreq}, "
          f"npol {obs.npol}")

    rf.FIXTURES.mkdir(parents=True, exist_ok=True)
    obs.save(str(rf.OBS_NPZ), True)
    digest = hashlib.sha256(rf.OBS_NPZ.read_bytes()).hexdigest()
    print(f"stage-extract: wrote {rf.OBS_NPZ} ({rf.OBS_NPZ.stat().st_size / 1e3:.0f} kB)")
    print(f"sha256 {digest}")
    print("Update CASA_OBS_SHA256 in test/conventions/conftest.py to this value.")


_STAGES = {
    "stage-model": stage_model,
    "stage-casa": stage_casa,
    "stage-extract": stage_extract,
}


def main() -> None:
    # The CASA launcher injects its own args around the script name, so pick
    # the stage token out of whatever argv we get.
    requested = [a for a in sys.argv if a in _STAGES]
    if not requested:
        print(__doc__)
        print("ERROR: name a stage: " + ", ".join(_STAGES))
        raise SystemExit(2)
    _STAGES[requested[0]]()


if __name__ == "__main__":
    main()
