# The sky frame — everything in one file

If you read nothing else in this directory, read this.  It answers one
question — *which way does a sky array point on the sky, how do we
know, and why was this ever hard?* — and every claim in it is paired
with a command you can run.  Nothing here rests on trusting the author;
that principle is itself the main lesson of this story.

Status: 2026-07-08.  Production code is at its original, validated
state.  The canonical frame below is an owner-approved TARGET; the
changes toward it (C1–C5) are planned, specified, and not yet made.

---

## 1. The question

A sky brightness distribution lives in a 2-D array `sky[i, j]`.  The
array itself has no North and no East — those meanings are assigned by
whatever code consumes it.  In this codebase, three consumers assign
them differently:

| consumer | reading of `sky[i, j]` | proof |
| --- | --- | --- |
| JWST response (interpolation path) | `i` = +Dec (North), `j` = −RA (West) | `p2_jwst_orientation.py` — a bump written North of center is read back by the sky point North of center, through astropy/gwcs |
| radio wgridder (ducc/finufft) | `i` = l-axis (RA direction), `j` = m-axis (Dec direction) | `p3_radio_orientation.py`, and externally `w7`/`w9` |
| `WcsAstropy` metadata (headers, `distances`) | claims `shape[0]` is the RA axis | `p1_metadata_vs_response.py` part A |

The **frame is a property of the sky-authoring side**: an array is
only "right" or "wrong" relative to which consumer reads it.  Every
bug in this story is a disagreement between an author and a consumer,
patched locally instead of being declared globally.

## 2. The target (owner-approved 2026-07-06; NOT yet enforced)

    sky[i, j]:  i = dim 0  ->  +Dec (North)     [row]
                j = dim 1  ->  -RA  (West)      [column]
    imshow(sky, origin="lower") renders North-up / East-left.

One frame for all authored skies; every consumer that needs another
layout converts EXPLICITLY at its own boundary, and every conversion
is pinned by a probe.

## 3. The history, in three acts

**Act I — the hidden reconciliation (until 2026-05-17).**  The shared
sky cube of the sptLensing joint fits was authored [Dec, RA] (the JWST
reading).  The radio extractor carried an unconditional
`resolve_transpose` converting it to the wgridder layout.  Everything
validated against literature (spt2147, spt0418) ran in this state.
The transpose was undocumented — a global toggle nobody remembered.

**Act II — the toggle flips (commit 737e517c, 2026-05-17).**
mosaic_imaging authored its skies natively in [RA, Dec]; for it the
transpose mirrored M51.  The fix removed the transpose *globally* —
correct for the mosaic, silently breaking the jwst+radio joint frame.
No production run happened after this, so the break stayed latent.
Root cause: an authoring-side property handled as shared-code state.

**Act III — the canonical attempt and the conjugation (2026-07-06/08).**
A canonical-frame fix was designed, probe-certified, and committed
(a5ab00a4).  Its radio adapter contained `conj(backend(sky.T))` — the
transpose was right, the conjugation was SPURIOUS.  Cause: the probe
that certified the adapter (the original p4) used a textbook analytic
anchor, `exp(−2πi(ul+vm))` at face value of the loaded uvw, which is
the CONJUGATE of what the CASA + `flip_v` pipeline realizes in those
variables.  Probe, implementation, and golden all descended from that
one wrong anchor — circular certification.  The roundtrip probe (p7)
could not catch it: it crosses the seam TWICE (forward + adjoint), and
the bilinear `jax.linear_transpose` adjoint applies a second
conjugation that cancels the first *exactly*.  The defect surfaced as
a "forward-then-dirty is rot180" test failure that had been filed as
an expected failure, was confirmed by a ONE-crossing discriminator
(`corr(conj(V_model), data) = 0.996` on the CASA fixture), reproduced
on the independent M51 dataset, and settled against the upstream
`resolve` package (whose forward has no conjugation).  Everything was
then reverted; production today is byte-identical to the last
validated state.

## 4. The measured facts (each with its anchor and its command)

| fact | anchor | run |
| --- | --- | --- |
| JWST path reads [Dec, RA] | astropy/gwcs | `p2`, `p6` |
| CASA data demand `B(truth.T)` — transpose yes, conjugation no (0.996 vs 0.064) | CASA | `p8_vis_seam_witness.py` |
| The import adds nothing: `\|vis − DATA\| = 0` exactly, no reorder, uvw untouched | casacore direct read | `w5` |
| The full stack recovers a CASA point source at 0.25" under the CURRENT contract; canonical authoring lands mirrored at 11.1" | CASA truth coordinates | `w7` |
| The radio chain is internally rectangle-consistent today; rectangle breakage lives at the WCS seams only | CASA, 96×128 grid | `w9`, then `p1` part C |
| Dirty imaging on the current code is anti-transpose = frame gap (C1) + non-Hermitian bilinear adjoint (C3) | CASA fixture | `p7` (documented failure) |
| Upstream resolve's forward = `vol · dirty2vis(sky, flip_v=True)`, no conj, no transpose | local copy `~/pro/python/resolve` | read `resolve/response.py` |

## 5. The epistemology (what this cost us to learn)

- **Goldens freeze behavior, not truth.**  A golden faithfully
  reproduces a founding error forever.
- **Roundtrips certify consistency, not correctness.**  Two crossings
  of a seam cancel involutions; conjugation is invisible to any
  forward-plus-adjoint image comparison — *mathematically*, not by bad
  luck.
- **In-repo analytic anchors certify agreement with their author.**
  If the implementation, the probe, and the golden share one anchor,
  no amount of internal cross-checking detects a wrong anchor.
- Therefore: **every seam needs a ONE-crossing witness against an
  EXTERNAL authority** (astropy/gwcs, CASA, casacore, upstream
  resolve), and the stack needs one end-to-end test in the direction
  inference actually uses it (w7).  And: **no xfail without owner
  sign-off** — an expected failure is a silenced finding.

## 6. The plan (details in `canonical-plan.md`)

- **C1** — radio response consumes canonical skies via ONE pure
  transpose (no conjugation, upstream-resolve parity).  Gates: p8's
  canonical row → ~0.99, w7's modes swap verdicts, p4 passes, w9's
  EXPECTED block updated in the same commit.
- **C2** — beams built canonically (geometry only).  Gate: p5.
- **C3** — dirty_image gets a Hermitian adjoint (conjugate the
  cotangent) — measured necessary by p7's rot180 residual.  Gate: p7
  verdict `identity` + the existing dirty golden byte-identical.
- **C4** — WcsAstropy metadata aligned to numpy order (rectangles
  correct at the WCS seams).  Ready on the `canonical-frame-wip`
  branch, with its rectangle-pinning probes.
- **C5** — downstream repos (mosaic_imaging re-authoring, plotting
  `.T` default, sptLensing FITS-export transpose), each gated.

Every change: owner runs the gate witnesses before and after; the
convention record (`check_frames.py` DOCUMENTED table) updates in the
same commit.

## 7. Where everything lives

- This directory: probes p1–p8, verifiers w5/w7/w9, fixtures in
  `golden/` (minted by CASA / jwst-datamodels; minting scripts in
  `roundtrip/`, needed only to re-mint).
- `check_frames.py` — run it any time: one verdict, including which
  failures are documented acceptance specs.
- `canonical-frame-wip` branch — the full previous attempt (batches
  D/E, the pytest sweep, rectangle acceptance tests), preserved.
- The reading path with exact expected numbers: `README.md`, "Guided
  read-through".

One sentence to remember, if only one survives: *the frame belongs to
the sky, conversions belong to boundaries, and no claim about either
counts until something outside this repository has confirmed it.*
