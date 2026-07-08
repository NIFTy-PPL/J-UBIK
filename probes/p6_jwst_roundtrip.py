"""p6 — JWST datamodel roundtrip through the real gwcs loader chain.

WHAT THIS PROBES
    The DATA-SIDE pairing of the real gwcs / datamodel path.  A synthetic
    JWST `ImageModel` (minted once by probes/roundtrip/mint_jwst_dm.py) has
    the orientation glyph (a letter "F"; probes/roundtrip/glyph.py) painted
    into its `data` array WORLD-ANCHORED through the model's own gwcs.  This
    probe loads that frozen FITS through the PRODUCTION chain

        JwstData(path)                                   # gwcs -> WcsJwstData
        -> bounding_indices_from_world_extrema(...)      # recon corners
        -> subsample_pixel_centers(bounds, jd.wcs, 1)    # data pixel centers
        -> world_coordinates_to_index_grid(..., "ij")    # -> recon (i, j)

    and scatters the data cutout onto a canonical reconstruction grid
    (dim0 = +Dec/North, dim1 = -RA/West; probes/README.md).  A correct
    loader chain lands the glyph unchanged: `dihedral_verdict` must return
    "identity".  A transposed data<->world pairing anywhere in the loader
    chain surfaces as a transpose-family verdict (transpose / anti-transpose
    / a rotation) — this probe is the witness that pins it down.

GOLDEN
    probes/golden/p6_scatter.npy — the scattered canonical image.  First
    run writes it; later runs assert byte-stable reproduction.  Its input,
    probes/golden/roundtrip_jwst_cal.fits, is minted separately and frozen.

RUN
    uv run python probes/p6_jwst_roundtrip.py
"""

import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # deterministic, device-independent

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.instruments.jwst.data.jwst_data import JwstData
from jubik.wcs import subsample_pixel_centers, world_coordinates_to_index_grid
from jubik.wcs.wcs_astropy import WcsAstropy

sys.path.insert(0, str(Path(__file__).parent / "roundtrip"))
import glyph  # noqa: E402
from mint_jwst_dm import (  # noqa: E402
    GLYPH_SCALE, RA0_DEG, DEC0_DEG, GOLDEN as MINTED_FITS,
)

GOLDEN = Path(__file__).parent / "golden" / "p6_scatter.npy"
DEFAULT_IMAGE = (
    Path(__file__).parent / "roundtrip" / "_images" / "p6_roundtrip.png"
)

RECON_SHAPE = (128, 128)
RECON_FOV = 16.0 * u.arcsec
RECON_PIXSIZE_ARCSEC = 0.125   # RECON_FOV / RECON_SHAPE


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
    center = SkyCoord(ra=RA0_DEG * u.deg, dec=DEC0_DEG * u.deg)

    # --- production loader chain ------------------------------------------
    jd = JwstData(str(MINTED_FITS))
    recon = WcsAstropy(center=center, shape=RECON_SHAPE,
                       fov=(RECON_FOV, RECON_FOV))

    bounds = jd.wcs.bounding_indices_from_world_extrema(recon.world_corners())
    min_row, max_row, min_col, max_col = bounds
    cutout = jd.dm.data[min_row:max_row, min_col:max_col]

    centers = subsample_pixel_centers(bounds, jd.wcs, subsample=1)
    idx = world_coordinates_to_index_grid([centers], recon, "ij")[0]

    # --- scatter the data onto the canonical grid -------------------------
    scattered = np.zeros(RECON_SHAPE)
    ii = np.round(idx[0]).astype(int)
    jj = np.round(idx[1]).astype(int)
    inb = (ii >= 0) & (ii < RECON_SHAPE[0]) & (jj >= 0) & (jj < RECON_SHAPE[1])
    np.add.at(scattered, (ii[inb], jj[inb]), cutout[inb])

    # --- truth: the glyph anchored at the recon pixel of the center -------
    x, y = recon.world_to_pixel(center)
    anchor = (int(round(float(y))), int(round(float(x))))  # canonical (i, j)
    truth = glyph.rasterize_canonical(
        RECON_SHAPE, anchor,
        (RECON_PIXSIZE_ARCSEC, RECON_PIXSIZE_ARCSEC),
        scale=GLYPH_SCALE,
    )

    verdict, scores = glyph.dihedral_verdict(scattered, truth)
    print(f"data cutout {cutout.shape} ({int((cutout > 0).sum())} lit) "
          f"-> scattered onto {RECON_SHAPE} ({int((scattered > 0).sum())} lit)")
    print(f"glyph anchor pixel (i, j) = {anchor}")
    print("\ndihedral scores (correlation vs each D4 transform of truth):")
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        mark = "  <-- verdict" if name == verdict else ""
        print(f"    {name:15s} {score:.4f}{mark}")
    print(f"\nDIHEDRAL VERDICT: {verdict}")

    assert verdict == "identity", (
        f"JWST loader chain roundtrip is NOT identity: got '{verdict}'. "
        "A transpose-family verdict means a transposed data<->world pairing "
        "somewhere in the gwcs/datamodel loader chain."
    )

    GOLDEN.parent.mkdir(exist_ok=True)
    if not GOLDEN.exists():
        np.save(GOLDEN, scattered)
        print(f"\ngolden WRITTEN: {GOLDEN.name}  shape={scattered.shape}")
    else:
        np.testing.assert_array_equal(scattered, np.load(GOLDEN))
        print(f"\ngolden REPRODUCED byte-identically: {GOLDEN.name}")

    print("\nVERDICT: synthetic JWST datamodel roundtrips through the real "
          "gwcs loader chain onto the canonical grid (dim0=+Dec, dim1=-RA).")

    if image_path is not None:
        _render_image(truth, scattered, "p6 — JWST roundtrip",
                      verdict, scores, image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JWST roundtrip probe")
    parser.add_argument(
        "--image", nargs="?", const=str(DEFAULT_IMAGE), default=None,
        metavar="PATH",
        help="after the golden check succeeds, render a truth-vs-roundtrip "
             f"PNG (default path: {DEFAULT_IMAGE})",
    )
    args = parser.parse_args()
    main(image_path=args.image)
