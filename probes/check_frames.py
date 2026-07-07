"""check_frames.py — THE consult-when-in-doubt script for sky-array frames.

Run this whenever you are unsure which way a sky array is oriented, or
whether the instrument boundaries still agree with the documented state:

    uv run python probes/check_frames.py

It runs every probe (p1..p7), compares the measured behavior
against the DOCUMENTED state below, and prints one consolidated verdict.  Green
means: the code still behaves exactly as documented here.  If this
script and the code ever disagree, trust the script's measurement and
update this file IN THE SAME COMMIT as any convention change — this
file is the convention record.

THE CANONICAL FRAME (decided 2026-07-06, owner-approved)
    sky[i, j]:  i = dim 0  ->  +Dec (North)     [row]
                j = dim 1  ->  -RA  (West)      [column]
    imshow(sky, origin="lower") renders North-up / East-left.
    All sky cubes and models at the jubik boundary are authored in this
    frame.  Instrument responses that need another layout must convert
    EXPLICITLY at their own boundary and be pinned by a probe here.

HISTORY (why this got confusing — do not repeat it)
    - The JWST response always read [Dec, RA] (probe p2).
    - The ducc/finufft wgridder natively reads [RA(l), Dec(m)] (probe p3).
    - Until commit 737e517c (2026-05-17) the radio extractor carried an
      unconditional resolve_transpose reconciling radio to [Dec, RA];
      all literature-validated sptLensing results (incl. the spt2147
      joint fit) ran WITH it.
    - mosaic_imaging authored its sky natively in [RA, Dec]; for it the
      transpose mirrored M51, so 737e517c REMOVED it globally — fixing
      the mosaic and silently breaking the joint jwst+radio frame.
    - Root cause: the frame was a property of the sky-authoring side,
      but the conversion was a hidden global toggle in shared code.
      The rule above (canonical frame + explicit per-boundary adapters,
      each pinned by a probe) is the permanent fix.
    - Batch A (2026-07-06) landed the explicit radio adapter
      (`canonical_sky_to_visibilities`, pinned by p4): the response now
      converts the canonical sky to the wgridder layout at its own boundary.
      The mosaic / sky_beamer path is still pending (Batch B).
    - Batch B (jubik side, 2026-07-07) landed the sky_beamer canonical
      frame: build_astropy_wcs / build_jft_sky_beamer now take numpy/
      canonical-ordered shape+fov and the sky_beamer transpose is gone, so
      beams pair index-for-index with the canonical sky (pinned by p5).
      What remains pending is re-authoring the mosaic_imaging PROJECT repo
      to canonical (its sky model + its own compensations).
    - Roundtrip probes (2026-07-07) landed: p6/p7 close the loop with
      externally-minted observations.  p6 paints the orientation glyph into
      a synthetic JWST datamodel world-anchored through its own gwcs and
      roundtrips it through the real loader chain onto the canonical grid;
      p7 does the same for a CASA-minted radio observation via a dirty
      image.  Their frozen inputs live in probes/golden/ and are re-minted
      only by probes/roundtrip/mint_*.py, never silently (see README).
"""

import subprocess
import sys
from pathlib import Path

PROBES_DIR = Path(__file__).parent
GOLDEN_DIR = PROBES_DIR / "golden"

# The documented state of every boundary.  Update in the same commit as
# any convention change.
DOCUMENTED = {
    "p1_metadata_vs_response.py": {
        "expect_pass": True,
        "meaning": "WcsAstropy metadata still pairs shape[0] with the RA "
                   "header axis while the JWST response reads dim0=Dec "
                   "(square-grid-only cancellation; non-square unsupported).",
    },
    "p2_jwst_orientation.py": {
        "expect_pass": True,
        "meaning": "JWST boundary COMPLIES with the canonical frame: "
                   "dim0=+Dec, dim1=-RA.",
    },
    "p3_radio_orientation.py": {
        "expect_pass": True,
        "meaning": "RAW gridder layer is wgridder-native by design: the "
                   "ducc/finufft backends read dim0=l/RA-axis, dim1=m/Dec-axis. "
                   "This is the low-level layout the response-level adapter "
                   "converts FROM — it is not the jubik boundary frame (see p4). "
                   "The golden pins this raw behaviour byte-identically.",
    },
    "p4_radio_adapter.py": {
        "expect_pass": True,
        "meaning": "Radio response COMPLIES with the canonical frame "
                   "(dim0=+Dec, dim1=-RA) via the explicit "
                   "canonical_sky_to_visibilities adapter wrapping the raw "
                   "gridder backends.",
    },
    "p5_sky_beamer_frame.py": {
        "expect_pass": True,
        "meaning": "Sky-beamer beams pair index-for-index with the canonical "
                   "sky (dim0=+Dec, dim1=-RA), pinned by the off-center-"
                   "pointing placement test.",
    },
    "p6_jwst_roundtrip.py": {
        "expect_pass": True,
        "meaning": "Synthetic JWST datamodel roundtrips through the real gwcs "
                   "loader chain onto the canonical grid (dim0=+Dec, dim1=-RA): "
                   "the world-anchored glyph scatters back to a dihedral "
                   "'identity' verdict. A transposed data<->world pairing in "
                   "the loader chain would show as a transpose-family verdict.",
        "requires": ["roundtrip_jwst_cal.fits"],
    },
    "p7_radio_roundtrip.py": {
        "expect_pass": True,
        "meaning": "CASA-minted observation roundtrips to a dirty image "
                   "matching the canonical truth (external sign-anchor pin): "
                   "the glyph survives the resolve response + gridder path with "
                   "a dihedral 'identity' verdict.",
        "requires": ["roundtrip_radio_obs.npz", "roundtrip_radio_truth.fits"],
    },
}


def main() -> int:
    print(__doc__)
    print("=" * 72)
    drifted = False
    for name, doc in DOCUMENTED.items():
        missing = [g for g in doc.get("requires", [])
                   if not (GOLDEN_DIR / g).exists()]
        if not (PROBES_DIR / name).exists():
            missing.append(name)
        if missing:
            print(f"\n{name}:  PENDING (golden not minted)")
            print(f"    {doc['meaning']}")
            print(f"    not yet present: {', '.join(missing)}")
            continue
        result = subprocess.run(
            [sys.executable, str(PROBES_DIR / name)],
            capture_output=True, text=True,
        )
        passed = result.returncode == 0
        ok = passed == doc["expect_pass"]
        drifted = drifted or not ok
        status = "AS DOCUMENTED" if ok else "DRIFTED FROM DOCUMENTATION"
        print(f"\n{name}:  {'pass' if passed else 'FAIL'}  ->  {status}")
        print(f"    {doc['meaning']}")
        if not ok:
            tail = "\n".join((result.stdout + result.stderr).splitlines()[-12:])
            print("    --- probe output (tail) ---")
            for line in tail.splitlines():
                print(f"    {line}")

    print("\n" + "=" * 72)
    if drifted:
        print("VERDICT: the code no longer matches the documented state.")
        print("Measure first (read the probe output above), then update the")
        print("DOCUMENTED table in the same commit as the code change.")
        return 1
    print("VERDICT: all boundaries behave exactly as documented above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
