# Convention probes & verifiers

**If you read one file, read `the-sky-frame.md`** — the whole story
(frames, history, measured facts, lessons, plan) in one place, every
claim paired with the command that proves it.

Status: 2026-07-08, after the trust reset.  **The canonical frame is a
PLAN, not a landed change** — all production code is at its original
(pre-canonical) state; everything in this directory is uncommitted.
The plan of record is `canonical-plan.md` (change set C1–C5, verifier
matrix W1–W8, binding process rules).  Nothing lands without the owner
running the relevant witness before and after.

## Guided read-through (read → run → verify)

Every step names what to read, what to run, and exactly what you must
see.  If any expectation fails, stop there — that is the finding.

**Step 1 — the plan.**  Read `canonical-plan.md` (5 min).  Verify for
yourself that the change set is small: C1 = one transpose, C2 = beam
geometry, C3 = one conjugated cotangent, and that NOTHING adds a
conjugation to the forward model.

**Step 2 — the consistency board.**  Run
`uv run python probes/check_frames.py`.  Expect: exit 0 and every
entry `AS DOCUMENTED` — including `p4`, `p5`, `p7` as documented
FAILURES (acceptance specs).  Read the docstring history while it runs
(~2 min): it contains the conjugation episode and why the goldens
missed it.

**Step 3 — the forward seam, your own eyes.**  Run
`uv run python probes/p8_vis_seam_witness.py`.  Expect three numbers:
`B(truth.T)` ≈ **0.996** (the validated-production composition),
`B(truth)` low, `conj(B(truth.T))` ≈ **0.064**.  The near-1 row is the
composition the CASA data demand: transpose yes, conjugation no.

**Step 4 — the whole stack, end to end.**  Run
`uv run python probes/w7_likelihood_position.py`.  Expect: gridder
mode recovers the source at (E +3.00", N +4.75") vs truth (3, 5) —
**0.25" error**; canonical mode REPORTED mirrored at **11.1"**; the
point-reflected-truth chi2 ~**20x** above the fitted chi2.  This is
the test class whose absence let the conjugation through.

**Step 5 — rectangles (where frame confusions stop cancelling).**  Run
`uv run python probes/w9_rectangle_radio_smoke.py`.  Expect: on a
96x128 grid the source lands at pixel **(42, 74)** exactly, reflected
and transposed hypotheses rejected at ~4.6x chi2 — the radio chain is
internally rectangle-consistent TODAY; the current code's rectangle
breakage is at the WCS seams, which `p1` part C demonstrates live
(run `uv run python probes/p1_metadata_vs_response.py` and read its
part C: world-span index ranges transposed against an (8,4) array).
Full rectangle acceptance tests (metadata, response-level, beamer)
exist on the `canonical-frame-wip` branch and land with C1/C4.

**Step 6 — the import seam.**  Run
`uv run python probes/w5_ms_import_fidelity.py`.  Expect:
`|vis - DATA| = 0` exactly while `|vis - conj(DATA)| = 1.124`, no row
reorder, uvw untouched — casacore itself certifying that `ms_import`
needs no change.

**Step 7 — judge.**  If steps 1–6 read true, the decision on the table
is C1 (one transpose), gated by: p8's canonical row going to ~0.99,
W7's two modes swapping verdicts, p4 passing, and w9's EXPECTED block
updated in the same commit.  If anything read false, the probes are
wrong, not the code — say so and we measure again.

## The target frame (the plan's statement — NOT yet enforced in code)

    sky[i, j]:  i = dim 0  ->  +Dec (North)     [row]
                j = dim 1  ->  -RA  (West)      [column]
    imshow(sky, origin="lower") renders North-up / East-left.

## Probe status against the CURRENT (reverted) code

| file | role | status |
| --- | --- | --- |
| `p1_metadata_vs_response.py` | measures the two metadata/response quirks of the current WcsAstropy (square-grid-only cancellation) | PASSES — true statement of current code (C4 on the wip branch replaces it when metadata alignment lands) |
| `p2_jwst_orientation.py` | JWST path reads [Dec, RA] — astropy-anchored | PASSES — was always true |
| `p3_radio_orientation.py` | raw gridder layout (dim0 = l-axis) vs an in-repo analytic anchor | PASSES — but see the caveat below on in-repo anchors |
| `p4_radio_adapter.py` | ACCEPTANCE SPEC for change C1 (canonical adapter = pure transpose, no conjugation; contract in the CASA-effective convention) | FAILS by design (ImportError) until C1 lands |
| `p5_sky_beamer_frame.py` | ACCEPTANCE SPEC for change C2 (canonical beams) | FAILS by design — measures the pre-canonical beam frame (peak (12,18) vs canonical (20,14)) |
| `p6_jwst_roundtrip.py` | JWST datamodel/gwcs loader roundtrip | PASSES |
| `p7_radio_roundtrip.py` | CASA dirty-image roundtrip | FAILS with verdict **anti-transpose** — the measured sum of the C1 frame gap (transpose) and the C3 non-Hermitian adjoint (rot180); becomes identity when C1+C3 land |
| `p8_vis_seam_witness.py` | W2 forward-seam witness (ONE crossing, CASA anchor), measurement-only | run it — no assertions |
| `w5_ms_import_fidelity.py` | W5: ms2observations vs direct casacore read, column by column | PASSES — import is conjugation-free, order-preserving, sign-untouched |
| `w7_likelihood_position.py` | W7: end-to-end likelihood position recovery, both frame modes | PASSES (gridder mode, the current contract: 0.25" error; canonical mode reported mirrored at 11.1" — the C1 gap as a number) |
| `w9_rectangle_radio_smoke.py` | rectangular sky (96×128) through the real response — rectangles are where frame confusions stop cancelling | PASSES — radio chain internally rectangle-consistent; the current code's rectangle breakage is at the WCS seams (p1 part C → change C4) |

## Goldens (`probes/golden/`)

Fixtures minted by external writers (CASA, jwst datamodels) — never
regenerated silently: `roundtrip_radio_obs.npz`, `roundtrip_radio_truth.fits`,
`roundtrip_jwst_cal.fits`, `w7_point_obs.npz`, `w7_truth.json`.
Probe-output freezes: the `p2/p3/p5/p6` `.npy` files (valid).
Two special cases, stated openly:
- `p4_adapter_vis.npy` was DELETED 2026-07-08: it froze the output of
  the (reverted) adapter under a wrong-signed analytic anchor — a
  golden pinning wrong physics.  C1's first passing run re-freezes it.
- `p7_dirty.npy` was frozen from the 2026-07-06 chain (transpose
  adapter + bilinear adjoint, whose two defects cancelled for CASA
  data).  The algebra says the corrected chain (C1 + C3) computes the
  same image; when C1+C3 land, p7 must reproduce it byte-identically —
  if it does not, that is a finding, not a formality.

## The lesson encoded here (why the W-verifiers exist)

An analytic anchor written in this repo certified a wrong convention
(the 2026-07-06/07 conjugation episode; full history in
`check_frames.py`).  Goldens freeze behavior, not truth; roundtrips
(two seam crossings) certify consistency, not correctness.  Therefore
every seam now has a ONE-crossing witness against an EXTERNAL
authority (astropy/gwcs, CASA, casacore, upstream resolve), and the
end-to-end W7 checks the whole stack in the direction inference
actually uses it.  No xfail without owner sign-off.

## Running

From the repo root (probes pin `JAX_PLATFORMS=cpu` themselves):

    uv run python probes/check_frames.py          # everything, one verdict
    uv run python probes/p8_vis_seam_witness.py   # the numbers, no claims
    uv run python probes/w7_likelihood_position.py

`--image` on p6/p7 renders a truth-vs-roundtrip PNG under
`probes/roundtrip/_images/`.  Minting scripts (CASA / jwst-datamodel)
live in `probes/roundtrip/`; they are needed only to re-mint fixtures,
never to verify.
