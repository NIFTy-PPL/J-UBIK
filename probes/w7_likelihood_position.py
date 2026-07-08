"""W7 -- END-TO-END likelihood position recovery: the decisive witness.

WHAT THIS CERTIFIES
    Fit REAL CASA-minted visibilities through the REAL jft Gaussian
    likelihood + the REAL jubik radio response, and check that the
    recovered source POSITION matches the truth in WORLD coordinates.

    This is the one test class whose absence let a conjugation error slip
    through the stack: a likelihood-level position fit exercises every seam
    at once, in the direction inference actually uses them.  A conjugation,
    mirror, rotation, or axis swap ANYWHERE (ms_import -> Observation ->
    interferometry_response -> wgridder -> the Gaussian likelihood) lands
    the fitted source at the WRONG world position.  A conjugation, in
    particular, point-reflects the source through the phase center -- so we
    report the chi2 of the point-reflected truth as its fingerprint.

EXTERNAL ANCHOR
    probes/golden/w7_point_obs.npz  -- CASA-simulated ALMA visibilities of
    a single 1 Jy point source at a KNOWN off-center offset.
    probes/golden/w7_truth.json     -- the truth in world coordinates:
    phase center J2000 13h37m00s -29d52m00s, offset East 3", North 5",
    minted by probes/roundtrip/mint_point_source_ms.py.  CASA owns the uvw
    geometry and visibility signs; jubik only reads the result back.  The
    offset is off-center AND East != North so a transpose (3,5)->(5,3) and a
    point reflection (3,5)->(-3,-5) are both visible.

THE TWO MODES (the authoring convention is a PARAMETER of this witness)
    The witness certifies whatever contract the code currently declares, so
    the [row, column] -> [sky axis] pairing is a mode, not a hard-coded
    assumption:

      mode="gridder"   -- sky array authored [RA, Dec] wgridder-native and
                          fed to interferometry_response AS-IS.  This is the
                          CURRENT (reverted) contract.  Expected to PASS.
      mode="canonical"  -- sky authored [Dec, RA] (dim0=+Dec, dim1=-RA), the
                          target frame AFTER change C1 lands.  On the current
                          as-is response it lands MIRRORED; reported, not
                          asserted.

    Both modes run the identical real forward model; they differ only in the
    (row, col) -> (East, North) frame used to read off the world position.
    The mode matching the code's current contract must recover the truth to
    within 0.5" (1 px); the other is reported to document the frame gap.

LIKELIHOOD
    The real path: jft.Gaussian(vis, noise_cov_inv=lambda x: x*weight)
    called on interferometry_response(obs, grid, Ducc0Settings(...))(sky).
    No hand-rolled chi2 -- jft.Gaussian(m) returns 0.5*sum(w*|m-vis|^2)
    directly (verified against the closed form).

OPTIMIZATION
    Deterministic, no stochastic samplers: a coarse full-grid search of the
    blob center (stride 2 px) at fixed flux, then a local Nelder-Mead
    refinement on (y0, x0, log-flux) around the best cell.

RUN (from the repo root)
    uv run python probes/w7_likelihood_position.py
"""

import json
import os
from pathlib import Path

# The jaxbind ducc kernels have no GPU FFI handler in this environment;
# pin the CPU platform before jax is imported (transitively, via jubik).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import astropy.units as u
import nifty.re as jft
import numpy as np
from astropy.coordinates import SkyCoord
from jax import jit
from jax import numpy as jnp
from scipy.optimize import minimize

from jubik.grid import Grid
from jubik.instruments.resolve.data import Observation
from jubik.instruments.resolve.parse.response import Ducc0Settings
from jubik.instruments.resolve.response import interferometry_response

GOLDEN_DIR = Path(__file__).parent / "golden"
OBS_NPZ = GOLDEN_DIR / "w7_point_obs.npz"
TRUTH_JSON = GOLDEN_DIR / "w7_truth.json"

# Reconstruction grid: 128 x 128 at 0.5"/px, centered on the phase center.
RECON_NPIX = 128
RECON_PIX_ARCSEC = 0.5
BLOB_SIGMA_PX = 0.7
COARSE_STRIDE = 2
CENTER_PX = RECON_NPIX // 2  # wgridder image center (l=m=0 = phase center)

# The response works in surface brightness: it integrates sky * pixel-solid-
# angle (vol = pixsize^2 in rad) to visibilities in Jy.  Fold vol into the
# blob normalization so the log-flux parameter is a physical total flux in Jy
# (otherwise the fit drives it to ~1/vol ~ 1e11 to match the data amplitude).
_ARCSEC2RAD = float(np.pi / 180.0 / 3600.0)
PIXEL_VOL_RAD2 = (RECON_PIX_ARCSEC * _ARCSEC2RAD) ** 2

# Tolerance for the mode that matches the code's current contract.
TOL_ARCSEC = 0.5


# === sky model =============================================================


def gaussian_blob(y0: float, x0: float, logflux: float) -> jnp.ndarray:
    """A normalized Gaussian blob at array pixel (y0, x0) with total flux e^logflux.

    Returns the 5-D sky (pol, time, freq, dim0, dim1) the response consumes.
    The array is authored in a fixed pixel layout; the [row, col] -> world
    frame is applied only afterwards, by the mode.
    """
    ii = jnp.arange(RECON_NPIX)
    di = ii[:, None] - y0
    dj = ii[None, :] - x0
    g = jnp.exp(-(di**2 + dj**2) / (2.0 * BLOB_SIGMA_PX**2))
    g = g * (jnp.exp(logflux) / (2.0 * jnp.pi * BLOB_SIGMA_PX**2 * PIXEL_VOL_RAD2))
    return g[None, None, None]


# === world-frame mode (row,col) -> (East, North) ===========================


def pixel_to_east_north(y0: float, x0: float, mode: str) -> tuple[float, float]:
    """Read the (East, North) arcsec offset of pixel (y0, x0) in the mode's frame.

    gridder  : array is [RA, Dec] wgridder-native -> East along dim0, North
               along dim1.
    canonical : array is [Dec, RA] (dim0=+Dec/North, dim1=-RA/West) -> North
               along dim0, East = -dim1.
    """
    dy = (y0 - CENTER_PX) * RECON_PIX_ARCSEC
    dx = (x0 - CENTER_PX) * RECON_PIX_ARCSEC
    if mode == "gridder":
        east = -dy
        north = dx
    elif mode == "canonical":
        east = -dx
        north = dy
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return float(east), float(north)


def east_north_to_pixel(east: float, north: float, mode: str) -> tuple[float, float]:
    """Inverse of pixel_to_east_north: where the mode's frame places (E, N)."""
    if mode == "gridder":
        y0 = CENTER_PX - east / RECON_PIX_ARCSEC
        x0 = CENTER_PX + north / RECON_PIX_ARCSEC
    elif mode == "canonical":
        y0 = CENTER_PX + north / RECON_PIX_ARCSEC
        x0 = CENTER_PX - east / RECON_PIX_ARCSEC
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return y0, x0


# === the fit ===============================================================


def run_mode(mode: str, chi2, truth: dict) -> None:
    """Fit the blob position, read it off in `mode`'s frame, print + assert."""
    east_t, north_t = truth["east_arcsec"], truth["north_arcsec"]

    # --- coarse full-grid search at fixed flux (log-flux 0 == truth flux) ---
    grid_axis = np.arange(0, RECON_NPIX, COARSE_STRIDE, dtype=float)
    best = (np.inf, CENTER_PX, CENTER_PX)
    for y0 in grid_axis:
        for x0 in grid_axis:
            val = float(chi2(y0, x0, 0.0))
            if val < best[0]:
                best = (val, y0, x0)
    _, cy, cx = best

    # --- local Nelder-Mead refinement on (y0, x0, log-flux) ----------------
    res = minimize(
        lambda p: float(chi2(p[0], p[1], p[2])),
        x0=np.array([cy, cx, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000},
    )
    y_fit, x_fit, logf_fit = res.x
    east_f, north_f = pixel_to_east_north(y_fit, x_fit, mode)
    err = float(np.hypot(east_f - east_t, north_f - north_t))

    # --- three-chi2 discrimination table -----------------------------------
    y_truth, x_truth = east_north_to_pixel(east_t, north_t, mode)
    y_refl, x_refl = 2 * CENTER_PX - y_truth, 2 * CENTER_PX - x_truth
    chi2_truth = float(chi2(y_truth, x_truth, 0.0))
    chi2_fit = float(chi2(y_fit, x_fit, logf_fit))
    chi2_refl = float(chi2(y_refl, x_refl, 0.0))

    print(f"\n=== mode = {mode!r} "
          f"({'array [RA, Dec], fed as-is' if mode == 'gridder' else 'array [Dec, RA]'}) "
          "===")
    print(f"  coarse-best cell (stride {COARSE_STRIDE}): "
          f"(y={cy:.0f}, x={cx:.0f}), chi2 {best[0]:.4e}")
    print(f"  refined pixel     : (y={y_fit:.3f}, x={x_fit:.3f}), "
          f"flux {np.exp(logf_fit):.4f} Jy")
    print(f"  fitted  (E, N)    : ({east_f:+.3f}, {north_f:+.3f}) arcsec")
    print(f"  truth   (E, N)    : ({east_t:+.3f}, {north_t:+.3f}) arcsec")
    print(f"  position error    : {err:.3f} arcsec  "
          f"({err / RECON_PIX_ARCSEC:.2f} px)")
    print("\n  chi2 discrimination (conjugation fingerprint):")
    print(f"    truth position           (y={y_truth:.1f}, x={x_truth:.1f}) : "
          f"{chi2_truth:.4e}")
    print(f"    fitted position          (y={y_fit:.1f}, x={x_fit:.1f}) : "
          f"{chi2_fit:.4e}")
    print(f"    point-reflected truth    (y={y_refl:.1f}, x={x_refl:.1f}) : "
          f"{chi2_refl:.4e}")

    if mode == "gridder":
        assert err <= TOL_ARCSEC, (
            f"mode={mode!r} is the code's CURRENT contract but recovered the "
            f"source {err:.3f}\" from truth (> {TOL_ARCSEC}\"). A conjugation, "
            f"mirror, rotation, or axis swap entered the stack -- compare the "
            f"three chi2 above (point-reflected << fitted => a conjugation)."
        )
        print(f"\n  VERDICT: mode={mode!r} recovers the truth position within "
              f"{TOL_ARCSEC}\" -- the current contract holds.")
    else:
        print(f"\n  REPORT (not asserted): mode={mode!r} lands {err:.3f}\" from "
              f"truth on the current as-is response -- it documents the frame "
              f"gap C1 closes, not a failure.")


def main() -> None:
    obs = Observation.load(str(OBS_NPZ))
    with open(TRUTH_JSON) as f:
        truth = json.load(f)

    center = SkyCoord(truth["phase_center_ra_hms"],
                      truth["phase_center_dec_dms"], frame="icrs")
    fov = [RECON_NPIX * RECON_PIX_ARCSEC] * 2 * u.arcsec
    grid = Grid.from_shape_and_fov(
        (RECON_NPIX, RECON_NPIX), fov, frequencies=None, sky_center=center,
    )

    print(f"observation : vis {obs.vis_val.shape}, npol {obs.npol}, "
          f"nfreq {obs.nfreq}, weight dtype {obs.weight_val.dtype}")
    print(f"grid        : {RECON_NPIX}x{RECON_NPIX} @ {RECON_PIX_ARCSEC}\"/px, "
          f"center px {CENTER_PX}, phase center {truth['phase_center']}")
    print(f"truth       : East {truth['east_arcsec']}\", "
          f"North {truth['north_arcsec']}\", flux {truth['flux_jy']} Jy")

    # --- the REAL response + REAL jft Gaussian likelihood ------------------
    R = interferometry_response(
        observation=obs,
        sky_grid=grid,
        backend_settings=Ducc0Settings(
            epsilon=1e-5, do_wgridding=False, nthreads=1, verbosity=0
        ),
    )
    vis = jnp.asarray(obs.vis_val)
    weight = jnp.asarray(obs.weight_val)
    likelihood = jft.Gaussian(vis, noise_cov_inv=lambda x: x * weight)

    @jit
    def chi2(y0, x0, logflux):
        return likelihood(R(gaussian_blob(y0, x0, logflux)))

    # both modes share the identical forward model; only the frame differs
    for mode in ("gridder", "canonical"):
        run_mode(mode, chi2, truth)

    print("\nW7 done: the mode matching the current contract recovered the "
          "truth world position; the other mode's offset documents the frame.")


if __name__ == "__main__":
    main()
