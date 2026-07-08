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

    CURRENT STATUS (measured 2026-07-08, on the reverted pre-canonical
    code): this probe FAILS with dihedral verdict ANTI-TRANSPOSE.  That
    verdict decomposes exactly into the two planned changes of
    probes/canonical-plan.md:
      - the TRANSPOSE part is the frame gap (the sky/dirty live in the
        wgridder layout, not the canonical one) — closed by change C1;
      - the residual ROT180 within the gridder's own frame means the
        dirty_image adjoint (jax.linear_transpose = the BILINEAR
        transpose) is not Hermitian — change C3, hereby MEASURED as
        necessary rather than hand-derived.
    Acceptance: after C1 + C3 this probe must report 'identity' AND
    reproduce probes/golden/p7_dirty.npy byte-identically (that golden
    was frozen from the 2026-07-06 chain whose two defects — a spurious
    adapter conjugation and the bilinear adjoint — cancelled for CASA
    data; the corrected chain computes the same image, so byte-identity
    is the algebraic cross-check).  A vis-domain seam stage (ONE forward
    crossing vs the CASA data; conjugation-proof) is part of the C1
    acceptance version of this probe; until then p8_vis_seam_witness.py
    covers that seam.  Do NOT patch production code from here; measure
    and report.

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

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from jubik.grid import Grid
from jubik.instruments.resolve.data import Observation
from jubik.instruments.resolve.dirty_image import dirty_image
from jubik.instruments.resolve.parse.response import Ducc0Settings

sys.path.insert(0, str(Path(__file__).parent / "roundtrip"))
from glyph import dihedral_verdict, rasterize_canonical  # noqa: E402

GOLDEN_DIR = Path(__file__).parent / "golden"
OBS_NPZ = GOLDEN_DIR / "roundtrip_radio_obs.npz"
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
        np.testing.assert_array_equal(img, np.load(DIRTY_GOLDEN))
        print(f"\ngolden REPRODUCED byte-identically: {DIRTY_GOLDEN.name}")

    print("\nVERDICT: RADIO roundtrip COMPLIES — CASA sky -> jubik dirty "
          "image is orientation-identity (sign anchor holds).")

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
