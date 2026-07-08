"""check_frames.py — THE consult-when-in-doubt script for sky-array frames.

Run this whenever you are unsure which way a sky array is oriented, or
whether the code still behaves exactly as documented here:

    uv run python probes/check_frames.py

It runs every probe/verifier, compares the measured behavior against
the DOCUMENTED state below, and prints one consolidated verdict.
Green means: the code still behaves exactly as documented — including
the failures that are documented as acceptance specs for planned
changes.  If this script and the code ever disagree, trust the
measurement and update this file IN THE SAME COMMIT as any convention
change — this file is the convention record.

THE TARGET FRAME (owner-approved 2026-07-06; a PLAN, not yet enforced)
    sky[i, j]:  i = dim 0  ->  +Dec (North)     [row]
                j = dim 1  ->  -RA  (West)      [column]
    imshow(sky, origin="lower") renders North-up / East-left.
    Plan of record: probes/canonical-plan.md (changes C1-C5, verifier
    matrix W1-W8, binding process rules).

HISTORY (why this got confusing — do not repeat it)
    - The JWST response always read [Dec, RA] (p2, astropy-anchored).
    - The ducc/finufft wgridder natively reads [RA(l), Dec(m)] (p3).
    - Until commit 737e517c (2026-05-17) the radio extractor carried an
      unconditional resolve_transpose reconciling radio to [Dec, RA];
      all literature-validated sptLensing results ran WITH it.
    - mosaic_imaging authored its sky natively in [RA, Dec]; for it the
      transpose mirrored M51, so 737e517c REMOVED it globally — fixing
      the mosaic and silently breaking the joint jwst+radio frame.
      Root cause: the frame was a property of the sky-authoring side,
      but the conversion was a hidden global toggle in shared code.
    - 2026-07-06: a canonical-frame fix was implemented and committed
      (a5ab00a4) — but its radio adapter carried a SPURIOUS CONJUGATION,
      caused by a wrong-signed analytic anchor written into the original
      p4.  The probe, the implementation, and the golden shared that one
      anchor (circular certification), and the image-domain roundtrip
      (p7) is mathematically blind to conjugation: the bilinear
      linear_transpose adjoint applies a second conjugation that cancels
      the first.  The defect was found by a ONE-crossing vis-domain
      discriminator (corr(conj(V_model), data) = 0.996 on the CASA
      fixture, confirmed on the independent M51 dataset and against the
      upstream resolve package), after surfacing first as a sweep xfail
      that had been framed away as "expected failure".
    - 2026-07-08: trust reset.  ALL canonical changes were removed from
      history (the owner reset to origin); production is at its
      original, validated state.  The full fixed state is preserved on
      the canonical-frame-wip branch.  Lessons made binding in
      canonical-plan.md: one-crossing external witnesses gate every
      change; goldens freeze behavior, not truth; roundtrips certify
      consistency, not correctness; no xfail without owner sign-off.

The DOCUMENTED table below states the MEASURED behavior of the CURRENT
(reverted, pre-canonical) code.  expect_pass=False entries are
acceptance specs for planned changes — their failure is the documented
truth of today's code, not a defect of the probes.
"""

import subprocess
import sys
from pathlib import Path

PROBES_DIR = Path(__file__).parent

# The documented state of every boundary.  Update in the same commit as
# any convention change.
DOCUMENTED = {
    "p1_metadata_vs_response.py": {
        "expect_pass": True,
        "meaning": "CURRENT WcsAstropy pairs shape[0] with the RA header "
                   "axis while the JWST response reads dim0=Dec — two "
                   "quirks cancelling on square grids only (change C4 "
                   "aligns the metadata; its rectangle-pinning p1 lives "
                   "on the wip branch).",
    },
    "p2_jwst_orientation.py": {
        "expect_pass": True,
        "meaning": "JWST boundary reads [Dec, RA] = the target canonical "
                   "frame; astropy-anchored; was always true.",
    },
    "p3_radio_orientation.py": {
        "expect_pass": True,
        "meaning": "RAW gridder layer is wgridder-native (dim0=l/RA-axis). "
                   "Caveat: judged against an IN-REPO analytic anchor — "
                   "the sign layer of such anchors is not trustworthy by "
                   "itself (see HISTORY); the external truth lives in "
                   "p8 and W7.",
    },
    "p4_radio_adapter.py": {
        "expect_pass": False,
        "meaning": "ACCEPTANCE SPEC for change C1 (canonical adapter = "
                   "pure transpose, NO conjugation, CASA-effective "
                   "convention).  Fails by design (ImportError) until C1 "
                   "lands; must pass afterwards.",
    },
    "p5_sky_beamer_frame.py": {
        "expect_pass": False,
        "meaning": "ACCEPTANCE SPEC for change C2 (canonical beams).  "
                   "Currently measures the pre-canonical beam frame "
                   "(peak (12,18) vs canonical (20,14)); must pass "
                   "after C2.",
    },
    "p6_jwst_roundtrip.py": {
        "expect_pass": True,
        "meaning": "JWST datamodel/gwcs loader roundtrips the glyph to a "
                   "dihedral 'identity' verdict on the canonical grid.",
    },
    "p7_radio_roundtrip.py": {
        "expect_pass": False,
        "meaning": "CASA dirty-image roundtrip: MEASURED verdict on the "
                   "current code is 'anti-transpose' = the C1 frame gap "
                   "(transpose) PLUS the C3 non-Hermitian bilinear "
                   "adjoint (rot180 in the gridder's own frame — C3 is "
                   "thereby measured as necessary, not derived).  Must "
                   "become 'identity' when C1+C3 land, with the existing "
                   "dirty golden reproduced byte-identically.",
    },
    "p8_vis_seam_witness.py": {
        "expect_pass": True,
        "meaning": "W2 forward-seam witness (ONE crossing, CASA anchor): "
                   "prints three correlations, asserts nothing — the "
                   "composition the data demand is the row near 1 "
                   "(currently B(truth.T), the validated-production "
                   "composition, no conjugation anywhere).",
    },
    "w5_ms_import_fidelity.py": {
        "expect_pass": True,
        "meaning": "W5: ms2observations certified column-by-column against "
                   "a direct casacore read — no conjugation (|vis-DATA|=0 "
                   "exactly), no row reorder, uvw sign untouched, npz "
                   "roundtrip byte-identical.  ms_import needs no change.",
    },
    "w7_likelihood_position.py": {
        "expect_pass": True,
        "meaning": "W7: end-to-end likelihood position recovery on CASA "
                   "data.  Current contract (gridder-frame authoring) "
                   "recovers the truth within 0.25 arcsec; canonical "
                   "authoring is REPORTED mirrored at 11.1 arcsec — the "
                   "C1 gap as a number.  After C1 the two modes must "
                   "swap verdicts.",
    },
    "w9_rectangle_radio_smoke.py": {
        "expect_pass": True,
        "meaning": "Rectangular sky (96x128) through the real response: "
                   "the CASA point source lands at the expected pixel "
                   "with reflected/transposed hypotheses rejected at "
                   "~4.6x chi2 — the radio chain is INTERNALLY "
                   "rectangle-consistent under the current contract; the "
                   "current code's rectangle breakage is at the WCS "
                   "seams (p1 part C), i.e. C4 territory.  Update its "
                   "EXPECTED block in the same commit as C1.",
    },
}


def main() -> int:
    print(__doc__)
    print("=" * 72)
    drifted = False
    for name, doc in DOCUMENTED.items():
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
    print("VERDICT: the code behaves exactly as documented above")
    print("(including the failures documented as pending acceptance specs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
