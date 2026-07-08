# The canonical-sky plan

Status: proposal, 2026-07-08.  Written after the full convention
investigation and the trust reset; every claim below is either backed
by an external witness the owner can run, or explicitly marked as
pending measurement.  Nothing lands without the owner running the
relevant witness before and after.

## Target statement

    sky[i, j]:  i = dim 0  ->  +Dec (North)     [row]
                j = dim 1  ->  -RA  (West)      [column]
    imshow(sky, origin="lower") renders North-up / East-left.

All sky cubes and models at the jubik boundary are authored in this
frame; instrument responses convert explicitly at their own boundary.

## Ground truths already measured (external anchors only)

1. The JWST path reads [Dec, RA] natively — anchored on astropy/gwcs
   (witnesses W1/W8; probes p2, p6).
2. The CASA data demand the composition `B(sky_gridder_layout)` with
   `sky_gridder_layout = canonical.T` and **no conjugation**:
   `corr(B(truth.T), data) = 0.996` on the roundtrip fixture, confirmed
   on the independent M51 dataset (three fields).  Run
   `probes/p8_vis_seam_witness.py` to see the three-row comparison
   yourself — it measures, asserts nothing.
3. This is exactly the composition all validated production ran
   (the `resolve_transpose` era fed `B(sky.T)`; mosaic_imaging authors
   `[RA, Dec]` skies = the same thing), and exactly upstream resolve's
   `SingleResponse` (`vol * dirty2vis(sky, flip_v=True)`, no conj —
   local copy at `~/pro/python/resolve`).
4. The 2026-07-06 adapter's conjugation was spurious (wrong-signed
   analytic anchor in the original p4); it is reverted and must not
   return.  Witness W2's control row pins its absence.

## Change set — best estimate of the REAL changes

**C1. Radio response consumes canonical skies via ONE pure transpose.**
`response.py`, inside `apply_R`: `r = op(inp)` becomes
`r = op(jnp.transpose(inp))` (or an equivalently named tiny helper).
No conjugation, no sign flips, raw builders untouched.  Equivalent to
upstream resolve fed a canonical sky.  Gate: W2 before (canonical row
low) and after (canonical row ≥ 0.99, conj row low).

**C2. Sky-beamer builds beams in the canonical frame.**
`sky_wcs.build_astropy_wcs` numpy-ordered; canonical mesh in
`build_jft_sky_beamer`; delete the `np.transpose(beam)` and its TODO.
Pure geometry — untouched by the conjugation saga.  Gate: W4
(off-center-pointing beam peak at the canonical pixel, all quadrants,
anisotropic pixels).

**C3. dirty_image — MEASURED AS NECESSARY (2026-07-08).**
On the reverted code, p7 fails with dihedral verdict ANTI-TRANSPOSE:
the transpose part is the C1 frame gap, and the residual rot180 within
the gridder's own frame is the `jax.linear_transpose` adjoint being
bilinear, not Hermitian.  Fix: conjugate the visibility cotangent
before the adjoint (`R^H(v) = R^T(conj(v))` for the C-linear C1
response).  Gate: W3 — p7 verdict 'identity' AND the existing
`p7_dirty.npy` golden reproduced byte-identically (the 2026-07-06
chain's two cancelling defects computed the same image the corrected
chain must compute; byte-identity is the algebraic cross-check).
Note this was measured, not hand-derived — hand-derivation is exactly
what failed at this seam before.

**C4. (Phase 2) WcsAstropy metadata alignment.**  shape/fov/distances
numpy-ordered, `world_corners`/`extent`/`from_wcs` fixed (the content of
the wip branch's Batch D).  Metadata-only; gate: W6 on rectangles plus
byte-identical W1–W5 outputs on squares.

**C5. (Phase 3, other repos, each with its own witness run)**
mosaic_imaging re-authored to canonical; radio plotting `.T` default
retired; sptLensing `save_fits` transpose retired.

**Explicitly NOT in the change set:** any conjugation anywhere (W2
forbids it); changes to the raw gridder builders (p3 pins them); changes
to `ms_import` (W5 certifies it as-is).

## Verifier matrix — every seam, external anchors, ins and outs

| id | seam (in -> out) | external anchor | status |
| --- | --- | --- | --- |
| W1 | sky array -> JWST data pairing | astropy WCS world/pixel | exists (p2) |
| W2 | sky array -> radio visibilities (ONE forward crossing) | CASA fixture + M51 | p8 measures; becomes an asserting witness with C1 |
| W3 | visibilities -> dirty image | CASA fixture + truth FITS (dihedral) | exists (p7); add forward-then-dirty identity, no xfail |
| W4 | sky array -> beamed sky (beam placement) | astropy WCS, off-center pointings | to write against C2 (canonical target) |
| W5 | MS on disk -> Observation (import fidelity) | casacore direct table read, column-by-column byte compare | DONE, passes (w5_ms_import_fidelity.py): no conjugation exactly, no reorder, uvw untouched |
| W6 | shape/fov/distances/header metadata | astropy WCS round-trips on rectangles | on wip branch (Batch D p1) |
| W7 | END-TO-END: CASA vis -> real jft likelihood -> fitted point-source position vs truth position | CASA + known truth coordinates | DONE, baseline recorded (w7_likelihood_position.py): gridder mode 0.25" from truth; canonical mode mirrored at 11.1" — the C1 gap as a number; the class of test whose absence let the conjugation through |
| W8 | datamodel FITS -> JwstData load | jwst/gwcs libraries | exists (p6) |

W7 is the decisive addition: a likelihood-level fit checks every seam
at once in the direction inference actually uses them — a conjugation,
mirror, rotation, or axis swap anywhere lands the fitted position in
the wrong place.

## Process rules (the lessons, made binding)

1. Every production change is gated by a ONE-crossing witness against
   an EXTERNAL reference, run by the owner before and after.
2. Roundtrips certify consistency, not correctness; goldens certify
   stability, not correctness; analytic anchors written in this repo
   certify nothing by themselves.
3. No xfail, ever, without owner sign-off — an expected failure is a
   silenced finding.
4. Commits are the owner's act.  Probes/witnesses are measurements and
   may be committed freely; production changes only after their gate.
