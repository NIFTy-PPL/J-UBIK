# Design of the canonical sky frame

Decided 2026-07-06. This page records what we decided about the orientation
of sky arrays and why. The API-level reference, with the property names and
migration notes, is [Spatial coordinate conventions](spatial-conventions.rst).

## The decision

One frame for every sky array at the jubik boundary. A field is stored as
`sky[..., dec, ra]`:

- dim 0 (rows) grows toward North, that is +Dec.
- dim 1 (columns) grows toward West, that is -RA.

Plotting it with `imshow(sky, origin="lower", extent=wcs.extent())` gives
North up and East left with no transpose. If you find yourself writing
`sky.T` before a plot, something upstream is wrong.

The public API speaks Cartesian order. `shape=(nx, ny)`,
`fov=(fov_x, fov_y)`, offsets as `(east, north)`. The same order appears in
YAML configs and in FITS headers (`CRPIX1` is the RA axis and has size `nx`).

The conversion between the two orders happens in exactly one place,
`SpatialGeometry.from_xy`. Everything else reads named accessors such as
`n_ra`, `n_dec`, `d_ra`, `d_dec`, `shape_yx`, or calls
`world_to_indices_yx` and `indices_yx_to_world`. No other code reverses a
tuple or indexes a shape with a literal.

An instrument backend that wants a different memory layout converts at its
own boundary, in one named function, and a test pins that function.

## Why we needed a decision

Until mid 2026 a single `Grid` had three effective layouts. `Grid.shape`
was `(nx, ny)`. The JWST response, the FITS writer and every plot assumed
`(row=Dec, col=RA)`. The radio gridders took `(l, m)` plus a `flip_v`
flag. Square grids with square pixels hide all of this, and most of our
reconstructions were square.

Commit 737e517c (2026-05-17) removed an unconditional transpose from the
radio extractor to fix the mosaic imaging pipeline, whose sky was authored in
`(RA, Dec)`. That fixed the mosaic and silently mirrored every joint
JWST plus radio reconstruction. Nobody noticed for weeks.

The root cause was structural, not a typo. The frame was a property of
whoever authored the sky, while the conversion lived as a hidden global
toggle in shared code. A fix for one author broke the others. The permanent
fix is a single frame and explicit conversions at named seams, each one
tested.

Why `(dec, ra)` internally? It is the order astropy hands us FITS data, and
the order `imshow` draws. Why `(x, y)` publicly? It is how people write a
shape or a field of view, and it matches the astropy pixel API and the FITS
header. The price is one conversion, so we gave that conversion one owner.

## The seams

| Boundary | What happens there | Pinned by |
| --- | --- | --- |
| `SpatialGeometry` | `from_xy` holds the only `[::-1]` in the package | metadata tests on rectangles and anisotropic pixels |
| JWST loader | gwcs pixel centers go through `world_to_indices_yx` onto the reconstruction grid; interpolators take yx indices only | orientation test and datamodel roundtrip |
| Resolve forward | `canonical_sky_to_visibilities` transposes to the wgridder `(l, m)` layout. Pure transpose, no conjugation. Emits `V = vol * exp(+2πi(ul + vm))` for uvw as `ms2observations` loads them, matching upstream `resolve` | adapter contract on both backends, CASA roundtrip |
| Resolve dirty image | `dirty_image` conjugates the visibility cotangent so the result is the Hermitian adjoint, not the bilinear transpose | adjoint identity test |
| Sky beamer | beams are built on the canonical grid and multiply the sky index for index | off-center pointing tests in all quadrants |
| FITS output | axis 1 is RA with size `nx`, axis 2 is Dec with size `ny` | header pairing tests |

Chandra and eROSITA accept the public `shape` but still require square
grids and reject rectangles. Their seams are not yet named.

## Where the tests live

Everything above is pinned under `test/conventions/`. The file names follow
the two kinds of claim on this page:

| File | Pins |
| --- | --- |
| `test_seam_spatial_geometry.py` | `SpatialGeometry` and `WcsAstropy`: XY in, YX storage, FITS axis pairing, `extent()`, `world_to_indices_yx` directions, `WcsAstropy_from_wcs` |
| `test_seam_radio_adapter.py` | `canonical_sky_to_visibilities` is exactly a transpose; raw gridders are RA-first; `dirty_image` is the Hermitian adjoint |
| `test_claims_jwst.py` | interpolation reads North up and East left; synthetic datamodel roundtrip is `identity` |
| `test_claims_radio.py` | point sources match the measurement equation on both backends and on a rectangle; glyph through `dirty_image` is `identity`; CASA roundtrip and vis-domain correlation |
| `test_claims_sky_beamer.py` | off-center pointings peak at the canonical pixel in all quadrants, square and rectangle |

Helpers: `glyph.py` (the F and the dihedral verdict), `radio_anchor.py`
(the measurement equation and stub observations), `jwst_fixture.py` (builds
the synthetic ImageModel at test time), `radio_fixture.py` (geometry of the
CASA run and the truth sky it used).

One data file is committed, `fixtures/roundtrip_radio_obs.npz`, the
CASA-simulated visibilities thinned to 8 integrations. Its sha256 is pinned
in `conftest.py`; a missing or altered file fails the run. It is re-minted
only by `mint/mint_radio_ms.py`, which needs a CASA install. The JWST
datamodel is not stored: `jwst_fixture` rebuilds it from `jwst` and `gwcs`
in about a second.

## How we know it holds

Two kinds of evidence, on purpose.

Analytic anchors. A unit point source at a known pixel must produce
visibilities matching the measurement equation, on anisotropic pixels so the
axis assignment is observable on a square grid. A beam pointed 4 arcsec North
and 4 arcsec East must peak at the pixel the frame predicts.

External witnesses. Analytic anchors written in this repo cannot catch a
convention that is wrong the same way on both sides. So we paint a letter F
onto the sky through a standard WCS, once into a synthetic JWST datamodel
through its own gwcs, once into a CASA-simulated measurement set. The F is
asymmetric under every rotation and reflection, so a frame error comes back
as a unique named element of the dihedral group. A correct roundtrip returns
`identity`, `rot180` means a sign layer is flipped, a transpose-family
verdict means an axis swap.

The CASA witness earned its place. The first radio adapter carried a
`jnp.conj` that came from an analytic anchor with the wrong sign. The dirty
image still came out `identity`, because the bilinear transpose in
`dirty_image` conjugated once more and the two errors cancelled. Meanwhile
the forward model was the complex conjugate of the data, and a likelihood
fit converged to the rotated sky. The radio roundtrip now also correlates
the forward model with the CASA visibilities directly, where no adjoint can
cancel anything.

## Rules when you touch this

1. Read `n_ra`, `n_dec`, `d_ra`, `d_dec` or `shape_yx`. Do not index a
   shape tuple.
2. No `.T`, `[::-1]`, `np.transpose` or flips on sky arrays outside
   `SpatialGeometry` and the named adapters.
3. A new instrument authors its sky in the canonical frame. If its backend
   wants another layout, write one named adapter and one test with an
   asymmetric glyph. Do not add a flag.
4. The CASA observation fixture is frozen and hash-pinned. Re-mint only
   when the Observation format or the glyph changes, in a commit that
   updates the hash and says so. Do not add stored output snapshots; every
   claim here has a direct assertion.
5. When the tests and this page disagree, trust the tests, then fix this
   page in the same commit.
