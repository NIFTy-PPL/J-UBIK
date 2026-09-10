# Convention probes

Executable witnesses for the spatial-grid convention at the instrument
boundaries.  Each probe is standalone, prints a verdict table, and
asserts it — running a probe **is** the verification act.

**In doubt about a frame?  Run `uv run python probes/check_frames.py`.**
It runs every probe, compares against the documented state of each
boundary, and prints one consolidated verdict.  Its docstring carries
the canonical frame *and* the history of how the conventions diverged
(the `resolve_transpose` / 737e517c story) — read that before touching
any orientation code.

## The normative convention (what everything is judged against)

Public geometry and coordinates use `(x, y)`:

    shape = (nx, ny)
    fov = (fov_x, fov_y)
    offsets = (x_east, y_north)  # Astropy quantities

Numerical fields use `sky[..., row, column] == sky[..., y, x]`. At
`position_angle=0`:

    i = dim 0 (row):    i increases  ->  Dec increases (North)
    j = dim 1 (column): j increases  ->  RA  decreases (West)

Plot the field itself—never `sky.T`—with
`imshow(sky, origin="lower", extent=wcs.extent())`. The extent is
`(+half_x, -half_x, -half_y, +half_y)`, so the result is **North up,
East left**. Rotated grids require WCSAxes; `extent()` rejects them.

This statement was measured on the shipped JWST response path
(2026-07-06); it is the *effective* behavior, produced by two
convention quirks that cancel on square grids (see `p1`).

## Probes

| probe | pins |
| --- | --- |
| `p1_metadata_vs_response.py` | public `shape_xy`/`fov_xy` are converted once to internal `shape_yx`/`fov_yx`; FITS, offsets, indices, East-left extent, rectangles, anisotropy, and Resolve metadata agree |
| `p2_jwst_orientation.py` | the normative convention above, on the shipped JWST interpolation path (point-source light-up from known sky directions) + golden freeze |
| `p3_radio_orientation.py` | which (axis, sign) mapping the resolve wgridder/finufft backends actually use for l/m, judged against analytic point-source visibilities; the raw wgridder-native layout the adapter converts from |
| `p4_radio_adapter.py` | the radio response COMPLIES with the canonical frame via the explicit `canonical_sky_to_visibilities` adapter (canonical sky in, physical visibilities out) wrapping the raw backends — a **pure axis transpose, conjugation-free**, emitting `V = vol·exp(+2πi(u·l + v·m))` in exact parity with upstream `resolve` |
| `p5_sky_beamer_frame.py` | the sky-beamer beams pair index-for-index with the canonical sky (dim0=+Dec, dim1=-RA): an off-center pointing on an anisotropic grid lands its beam peak at the canonical sky pixel of the pointing direction |
| `p6_jwst_roundtrip.py` | a synthetic JWST datamodel (glyph painted world-anchored through its own gwcs) roundtrips through the production loader chain and `world_to_indices_yx()` onto the canonical grid with a dihedral `identity` verdict |
| `p7_radio_roundtrip.py` | a CASA-minted radio observation roundtrips to a dirty image matching the canonical truth (external sign-anchor pin): the glyph survives the resolve response + gridder path with a dihedral `identity` verdict, **and** the forward model matches the CASA visibilities directly in the vis domain (`corr(V_model, data) > 0.99`) so no adjoint-side cancellation can mask a spurious conjugation |

## Golden freeze

Probes write small witness arrays to `probes/golden/` on first run and compare
against them thereafter with tight, probe-specific numerical tolerances. The
golden files themselves are immutable: convention changes must keep their
checked-in bytes unchanged unless a separately reviewed re-mint is intended.

## Running

From the repo root:

    uv run python probes/p1_metadata_vs_response.py
    uv run python probes/p2_jwst_orientation.py
    uv run python probes/p3_radio_orientation.py
    uv run python probes/p4_radio_adapter.py
    uv run python probes/p5_sky_beamer_frame.py
    uv run python probes/p6_jwst_roundtrip.py
    uv run python probes/p7_radio_roundtrip.py

Pass `--image` (optionally a path) to `p6`/`p7` to render a
truth-vs-roundtrip PNG under `probes/roundtrip/_images/` after the golden
check passes; the flag-less run is unchanged.

## Roundtrip goldens

The roundtrip probes (`p6`, `p7`) close the frame loop against
EXTERNALLY-MINTED observations rather than analytic stand-ins: an
orientation glyph is painted into a real instrument data product, carried
through the actual production loader/response chain, and checked back
against the canonical truth by `dihedral_verdict`.  Their frozen inputs
live in `probes/golden/` and are produced by one-time minting scripts:

| golden input | minted by | consumed by |
| --- | --- | --- |
| `roundtrip_jwst_cal.fits` | `probes/roundtrip/mint_jwst_dm.py` | `p6_jwst_roundtrip.py` |
| `roundtrip_radio_obs.npz`, `roundtrip_radio_truth.fits` | `probes/roundtrip/mint_radio_ms.py` | `p7_radio_roundtrip.py` |

Re-run a minting script **only** if the observation format (datamodel /
gwcs / measurement-set layout) or the glyph itself changes — **never
silently**, because regenerating rewrites the frozen input the probe pins
against.  Each mint script's header docstring carries the single command
to run it (in the j-ubik venv).  `check_frames.py` treats a probe whose
golden input has not yet been minted as `PENDING`, not `FAIL`.

## Pytest sweep

    uv run pytest test/conventions -q

The probes remain the human-facing consult record; `test/conventions/` is
their pytest twin.  It runs the probe record (every probe p1..p7 as a
subprocess plus `check_frames.py`, exit-code asserted; p6/p7 skip when their
goldens are not minted) *and* a parametrized breadth sweep the single-witness
probes do not cover: public-XY rectangular and both-orientation grids, isotropic vs 2:1
anisotropic pixels, both radio backends (ducc + finufft), all four pointing
quadrants, a response-level rectangle pin (`npix_x = shape_xy[0]` /
`pixsize_x = pixel_scales_xy[0]`), and the dirty-image adjoint identity. It is
golden-free except where the probe-record tests reuse the existing frozen
goldens read-only.
