"""p7 — the RADIO roundtrip SIGN ANCHOR: CASA sky in, jubik dirty image out.

WHAT THIS PROBES
    Every other radio probe (p3/p4) judges the response against an
    analytic measurement equation written in THIS repo — a self-consistent
    anchor, but one that cannot catch a convention that is wrong the same
    way on both sides.  p7 closes that loop with an EXTERNAL witness: the
    fixture visibilities were minted by CASA (probes/roundtrip/mint_radio_ms.py)
    from a truth sky painted through a standard FITS RA/Dec WCS.  CASA owns
    the uvw geometry and the visibility sign convention; jubik only reads
    the result back.

    So a wrong uvw / visibility-conjugation / axis convention ANYWHERE in
    the jubik radio path (ms2observations -> Observation -> the
    canonical_sky_to_visibilities adapter -> the ducc wgridder -> the
    dirty_image linear_transpose adjoint) can no longer hide: it shows up
    here as a non-identity dihedral verdict on the dirty image.  The F glyph
    is asymmetric under all of D4, so the error is uniquely named:

        identity        -> the whole chain is consistent (correct)
        rot180          -> a SIGN layer is flipped (uv or vis conjugation)
        transpose family-> an AXIS swap (l<->m / dim0<->dim1)
        flip-dim0/1     -> a single-axis sign flip

    HISTORY — why the VIS-DOMAIN stage below exists (2026-07-07): the
    dihedral verdict alone once hid a real defect.  Batch A's adapter
    carried a spurious jnp.conj (a wrong-signed analytic anchor in p3/p4),
    and dirty_image's jax.linear_transpose is the BILINEAR transpose whose
    implicit conjugation cancelled it — two wrongs made the dirty image
    come out "identity" while the FORWARD model was the complex conjugate
    of the data (measured: corr(V_model, d) = 0.064, corr(conj(V_model),
    d) = 0.996; confirmed independently on the M51 dataset).  A likelihood
    fit against that seam converges to the rot180 sky.  The fix (Batch E)
    dropped the conj — restoring parity with the upstream `resolve`
    package (vol * dirty2vis(sky, flip_v=True), no conj) — and made
    dirty_image Hermitian.  The vis-domain stage pins the seam DIRECTLY
    so no adjoint-side cancellation can ever mask it again.

    ENVIRONMENT: forces the JAX CPU platform (the jaxbind ducc kernels have
    no GPU FFI handler in this env) and uses gridder epsilon 1e-5 because
    the CASA fixture carries single-precision weights (float32 has no 1e-9
    ducc kernel).  Both are probe-side backend choices, not jubik changes.

GOLDEN
    probes/golden/p7_dirty.npy — the dirty image.  First run writes, later
    runs assert byte-stable reproduction (write-once / reproduce, like p4/p5).
    The fixture itself (roundtrip_radio_obs.npz + roundtrip_radio_truth.fits)
    is minted once by mint_radio_ms.py and is not regenerated here.

RUN
    uv run python probes/p7_radio_roundtrip.py
"""

import os
import sys
from pathlib import Path

# The jaxbind ducc kernels have no GPU FFI handler in this environment;
# pin the CPU platform before jax is imported (transitively, via jubik).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from jubik.grid import Grid
from jubik.instruments.resolve.data import Observation
from jubik.instruments.resolve.dirty_image import dirty_image
from jubik.instruments.resolve.parse.response import Ducc0Settings
from jubik.instruments.resolve.response import (
    canonical_sky_to_visibilities,
    interferometry_response_ducc,
)

sys.path.insert(0, str(Path(__file__).parent / "roundtrip"))
from glyph import dihedral_verdict, rasterize_canonical  # noqa: E402

GOLDEN_DIR = Path(__file__).parent / "golden"
OBS_NPZ = GOLDEN_DIR / "roundtrip_radio_obs.npz"
TRUTH_FITS = GOLDEN_DIR / "roundtrip_radio_truth.fits"
DIRTY_GOLDEN = GOLDEN_DIR / "p7_dirty.npy"
DEFAULT_IMAGE = (
    Path(__file__).parent / "roundtrip" / "_images" / "p7_roundtrip.png"
)

# Must match the mint (mint_radio_ms.py): phase center + field extent.
CENTER = SkyCoord("13h37m00s", "-29d52m00s", frame="icrs")
RECON_NPIX = 128
RECON_PIX_ARCSEC = 0.5           # 128 * 0.5" = 64" == truth field (256 * 0.25")
FOV = [RECON_NPIX * RECON_PIX_ARCSEC] * 2 * u.arcsec
GLYPH_SCALE = 1.0                # same glyph scale the mint painted


def _draw_compass(ax) -> None:
    """Small compass (N up, E left) in axes-fraction coords."""
    arrow = dict(arrowstyle="->", color="red", lw=1.6)
    ax.annotate("", xy=(0.12, 0.93), xytext=(0.12, 0.73),
                xycoords="axes fraction", arrowprops=arrow)
    ax.text(0.12, 0.97, "N", transform=ax.transAxes, color="red",
            fontsize=11, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(0.05, 0.80), xytext=(0.25, 0.80),
                xycoords="axes fraction", arrowprops=arrow)
    ax.text(0.02, 0.80, "E", transform=ax.transAxes, color="red",
            fontsize=11, fontweight="bold", ha="left", va="center")


def _render_image(truth: np.ndarray, roundtrip: np.ndarray, probe_name: str,
                  verdict: str, scores: dict, out_path: str) -> Path:
    """Save a side-by-side truth-vs-roundtrip PNG with a compass overlay."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend, set before pyplot import
    import matplotlib.pyplot as plt

    runner_up = max(v for k, v in scores.items() if k != verdict)
    margin = scores[verdict] - runner_up

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, arr, title in (
        (axes[0], truth, "canonical truth glyph"),
        (axes[1], roundtrip, "roundtrip image"),
    ):
        ax.imshow(arr, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("dim 1  (−RA → West right, East left)")
        ax.set_ylabel("dim 0  (+Dec → North up)")
        _draw_compass(ax)

    fig.suptitle(f"{probe_name}    dihedral verdict: {verdict} "
                 f"(margin {margin:+.3f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nimage WRITTEN: {out}")
    return out


def main(image_path: str | None = None) -> None:
    obs = Observation.load(str(OBS_NPZ))
    print(f"observation: vis {obs.vis_val.shape}, npol {obs.npol}, "
          f"nfreq {obs.nfreq}, weight dtype {obs.weight_val.dtype}")

    grid = Grid.from_shape_and_fov(
        (RECON_NPIX, RECON_NPIX), FOV, frequencies=None, sky_center=CENTER,
    )

    dirty = dirty_image(
        obs, grid,
        Ducc0Settings(epsilon=1e-5, do_wgridding=False, nthreads=1, verbosity=0),
        weighting="natural",
    )
    img = np.real(np.asarray(dirty.value))[0, 0, 0]
    print(f"dirty image: shape {img.shape}, peak {img.max():.3f} at "
          f"{np.unravel_index(int(np.argmax(img)), img.shape)}")

    # Canonical truth at the reconstruction pixel scale (dim0=+Dec, dim1=-RA).
    # dihedral_verdict is shift-invariant, so the glyph anchor need not match.
    truth = rasterize_canonical(
        (RECON_NPIX, RECON_NPIX),
        (RECON_NPIX // 2, RECON_NPIX // 2),
        (RECON_PIX_ARCSEC, RECON_PIX_ARCSEC),
        scale=GLYPH_SCALE,
    )

    verdict, scores = dihedral_verdict(img, truth)
    runner_up = max(v for k, v in scores.items() if k != verdict)
    print("\n  D4 transform     correlation")
    print("  " + "-" * 34)
    for name in sorted(scores, key=scores.get, reverse=True):
        mark = "  <-- winner" if name == verdict else ""
        print(f"  {name:16s} {scores[name]:.4f}{mark}")
    print(f"\n  VERDICT: {verdict}  (margin over runner-up "
          f"{scores[verdict] - runner_up:+.4f})")

    assert verdict == "identity", (
        f"roundtrip dihedral verdict is {verdict!r}, not 'identity' — the "
        f"CASA-minted sign anchor disagrees with the jubik radio chain. "
        f"rot180 => a visibility/uv sign layer; transpose family => an axis "
        f"swap. This is a real convention finding: measure, do not patch "
        f"production code from the probe."
    )

    # --- golden freeze (write-once / reproduce) -----------------------
    GOLDEN_DIR.mkdir(exist_ok=True)
    if not DIRTY_GOLDEN.exists():
        np.save(DIRTY_GOLDEN, img)
        print(f"\ngolden WRITTEN: {DIRTY_GOLDEN.name}")
    else:
        np.testing.assert_allclose(
            img, np.load(DIRTY_GOLDEN), rtol=1e-5, atol=2e-4
        )
        print(f"\ngolden matched within numerical tolerance: {DIRTY_GOLDEN.name}")

    # --- VIS-DOMAIN seam stage (the test the dirty image cannot do) ----
    # Forward-model the CASA truth sky through the shipped adapter and
    # correlate with the CASA visibilities DIRECTLY.  A model/data
    # conjugation cannot hide here behind any adjoint-side cancellation.
    with fits.open(TRUTH_FITS) as hdul:
        truth_native = np.squeeze(hdul[0].data).astype(np.float64)
        dpix_rad = abs(hdul[0].header["CDELT1"]) * np.pi / 180.0
    stub = SimpleNamespace(uvw=np.asarray(obs.uvw), freq=np.asarray(obs.freq))
    backend = interferometry_response_ducc(
        stub, npix_x=truth_native.shape[1], npix_y=truth_native.shape[0],
        pixsize_x=dpix_rad, pixsize_y=dpix_rad,
        do_wgridding=False, epsilon=1e-5, nthreads=1, verbosity=0,
    )
    v_model = np.asarray(
        canonical_sky_to_visibilities(lambda s: backend(s), truth_native)
    ).ravel()
    d = np.asarray(obs.vis_val[0]).ravel()

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(np.vdot(a, b))
                     / (np.linalg.norm(a) * np.linalg.norm(b)))

    direct, conjugated = _corr(v_model, d), _corr(np.conj(v_model), d)
    print(f"\nvis-domain seam: corr(V_model, data) = {direct:.4f}, "
          f"corr(conj(V_model), data) = {conjugated:.4f}")
    assert direct > 0.99, (
        f"forward model does not match the CASA visibilities directly "
        f"(corr {direct:.3f}); if the CONJUGATED correlation is high "
        f"({conjugated:.3f}) the response carries a spurious conjugation "
        f"— a likelihood fit would converge to the rot180 sky."
    )
    assert conjugated < 0.5, "conjugated model also correlates — degenerate?"

    print("\nVERDICT: RADIO roundtrip COMPLIES — CASA sky -> jubik dirty "
          "image is orientation-identity AND the forward model matches "
          "the CASA visibilities directly (seam conjugation-free).")

    if image_path is not None:
        _render_image(truth, img, "p7 — RADIO roundtrip",
                      verdict, scores, image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RADIO roundtrip probe")
    parser.add_argument(
        "--image", nargs="?", const=str(DEFAULT_IMAGE), default=None,
        metavar="PATH",
        help="after the golden check succeeds, render a truth-vs-roundtrip "
             f"PNG (default path: {DEFAULT_IMAGE})",
    )
    args = parser.parse_args()
    main(image_path=args.image)
