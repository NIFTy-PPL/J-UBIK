"""W5 — MS import fidelity: MS on disk -> jubik Observation, column by column.

WHAT THIS CERTIFIES
    The seam `ms2observations(ms, "DATA", ...) -> Observation` is a faithful
    transcription of the measurement set: it neither CONJUGATES the
    visibilities, nor REORDERS the rows, nor FLIPS the sign of the uvw
    coordinates, nor drops sign information from any measured column.  This is
    the import half of the radio path — every downstream orientation probe
    (p3/p4/p7/p8/W2/W3) trusts that what jubik hands it equals what CASA wrote.

EXTERNAL ANCHOR
    casacore.tables — a DIRECT, independent read of the very same MS tables
    (DATA, UVW, ANTENNA1/2, TIME, FLAG, WEIGHT/WEIGHT_SPECTRUM, SPECTRAL_WINDOW
    CHAN_FREQ, FIELD PHASE_DIR/REFERENCE_DIR).  Nothing here is judged against
    a formula written in this repo; casacore itself is the reference.  jubik's
    reader (`read_ms_i`) and this witness both call `casacore.tables.table`,
    so any disagreement is a transformation jubik applied on top of the raw
    columns — exactly what we want to expose.

STYLE
    Measurement-first.  Every comparison prints its max-abs-diff / equality
    number; the asserts at the end pin only the involution-critical facts
    (no conjugation, no row reorder, uvw sign untouched, npz round trip exact).

KNOWN LAYOUT (verified on the fixture)
    MS DATA column ...... (nrow, nchan, npol)
    jubik vis_val ....... (npol, nrow, nchan)   via _ms2resolve_transpose (2,0,1)
    So  vis_val[p, r, c] == DATA[r, c, p]  with jubik's rule vis[weight==0]=0
    applied.  The polarization axis order is the MS CORR_TYPE order.

RUN (from the repo root)
    uv run python probes/w5_ms_import_fidelity.py
"""

import os
import tempfile
from pathlib import Path

# The resolve stack pulls in jax/jaxbind ducc kernels with no GPU FFI handler
# in this environment; pin the CPU platform before jax imports transitively.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from casacore.tables import table

from jubik.instruments.resolve.data.ms_import import ms2observations

# --- the MS under test -------------------------------------------------------
PRIMARY_MS = (
    Path(__file__).parent
    / "roundtrip" / "_casa_work" / "roundtrip" / "roundtrip.alma.cycle1.1.ms"
)
# Fallback if the (gitignored) CASA sim has been cleaned.
FALLBACK_MS = Path(
    "/home/jruestig/pro/python/radio_projects/mosaic_imaging/data/"
    "m51_synthetic/m51c/m51c.ALMA_0.5arcsec.ms"
)

FIELD = 0
SPW = 0

SCRATCH = Path(
    "/tmp/claude-1000/-home-jruestig-pro-python-charm-lensing-NewStructureReference/"
    "8e690556-8e6a-4d8e-97e4-3aed45038555/scratchpad"
)


def _resolve_ms() -> tuple[Path, str]:
    if PRIMARY_MS.exists():
        return PRIMARY_MS, "primary (CASA roundtrip sim)"
    if FALLBACK_MS.exists():
        return FALLBACK_MS, "FALLBACK (M51 synthetic — primary MS was cleaned)"
    raise FileNotFoundError(
        f"neither the primary MS ({PRIMARY_MS}) nor the fallback "
        f"({FALLBACK_MS}) exists"
    )


def _open(ms: str, sub: str = ""):
    path = ms if not sub else os.path.join(ms, sub)
    return table(path, readonly=True, ack=False)


def _row_max_absdiff(a: np.ndarray, b: np.ndarray) -> float:
    """Max abs difference, treating exact-equal as 0.0 for clean printing."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> None:
    ms_path, provenance = _resolve_ms()
    ms = str(ms_path)
    print("=" * 74)
    print("W5 — MS import fidelity (external anchor: casacore direct table read)")
    print("=" * 74)
    print(f"MS       : {ms}")
    print(f"provenance: {provenance}")
    print(f"field={FIELD}  spw={SPW}\n")

    # --- (1) DIRECT casacore read -------------------------------------------
    with _open(ms) as t:
        field_id = t.getcol("FIELD_ID")
        spw_of_row = t.getcol("DATA_DESC_ID")
        raw_data = t.getcol("DATA")            # (nrow, nchan, npol) complex64
        raw_flag = t.getcol("FLAG")            # (nrow, nchan, npol) bool
        raw_uvw_all = t.getcol("UVW")          # (nrow, 3)
        raw_ant1_all = t.getcol("ANTENNA1")
        raw_ant2_all = t.getcol("ANTENNA2")
        raw_time_all = t.getcol("TIME")
        colnames = t.colnames()
        full_weight = t.getcol("WEIGHT")       # (nrow, npol)
        has_wspec = "WEIGHT_SPECTRUM" in colnames

    with _open(ms, "POLARIZATION") as t:
        corr_type = t.getcol("CORR_TYPE")[0]   # e.g. [9, 12] == [XX, YY]
    with _open(ms, "SPECTRAL_WINDOW") as t:
        chan_freq = t.getcol("CHAN_FREQ")[SPW]  # (nchan,)
    with _open(ms, "FIELD") as t:
        phase_dir = t.getcol("PHASE_DIR")[FIELD][0]      # (2,) rad
        reference_dir = t.getcol("REFERENCE_DIR")[FIELD][0]

    if has_wspec:
        with _open(ms) as t:
            try:
                raw_wspec = t.getcol("WEIGHT_SPECTRUM")  # (nrow, nchan, npol)
            except RuntimeError:
                raw_wspec = None
    else:
        raw_wspec = None

    POL_NAMES = {5: "RR", 6: "RL", 7: "LR", 8: "LL",
                 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
    pol_labels = [POL_NAMES.get(int(c), str(int(c))) for c in corr_type]

    # jubik's active-row selection keeps, in original order, every row of the
    # requested field/spw that is not fully flagged.  Replicate that mask from
    # the raw columns so the direct read aligns row-for-row with jubik.
    row_flagged = raw_flag.all(axis=(1, 2))          # all pol & chan flagged
    sel = (field_id == FIELD) & (spw_of_row == SPW) & (~row_flagged)
    raw_uvw = np.ascontiguousarray(raw_uvw_all[sel])
    raw_ant1 = np.ascontiguousarray(raw_ant1_all[sel])
    raw_ant2 = np.ascontiguousarray(raw_ant2_all[sel])
    raw_time = np.ascontiguousarray(raw_time_all[sel])
    raw_data_sel = raw_data[sel]                     # (nsel, nchan, npol)
    raw_flag_sel = raw_flag[sel]
    raw_weight_sel = full_weight[sel]                # (nsel, npol)

    print(f"MS columns present    : {len(colnames)} cols; "
          f"WEIGHT_SPECTRUM={'yes' if has_wspec else 'no (using WEIGHT)'}")
    print(f"CORR_TYPE (pol order) : {list(map(int, corr_type))} -> {pol_labels}")
    print(f"rows total / selected : {raw_data.shape[0]} / {sel.sum()} "
          f"(field/spw/unflagged)")
    print(f"any FLAG set          : {bool(raw_flag.any())}  "
          f"(frac {raw_flag.mean():.4f})")
    print(f"DATA shape (r,c,p)    : {raw_data.shape}")
    print()

    # --- (2) jubik read ------------------------------------------------------
    obs_list = ms2observations(ms, "DATA", True, SPW)
    obs = [o for o in obs_list if o is not None]
    # Pick the observation of the requested field (ms2observations returns one
    # entry per field, None for empty fields).
    assert obs, "ms2observations returned no non-empty observation"
    obs = obs[FIELD] if obs_list[FIELD] is not None else obs[0]

    jub_vis = np.asarray(obs.vis_val)        # (npol, nrow, nchan)
    jub_wgt = np.asarray(obs.weight_val)
    jub_uvw = np.asarray(obs.uvw)
    jub_ant1 = np.asarray(obs.ant1)
    jub_ant2 = np.asarray(obs.ant2)
    jub_time = np.asarray(obs.time)
    jub_freq = np.asarray(obs.freq)
    jub_pol = obs.legacy_polarization.to_str_list()

    print(f"jubik vis_val shape   : {jub_vis.shape} (npol,nrow,nchan) "
          f"{jub_vis.dtype}")
    print(f"jubik pol labels      : {jub_pol}")
    print()

    # --- (3) column-by-column comparison ------------------------------------
    # Build the raw column in jubik's (npol, nrow, nchan) layout for a direct
    # element-wise compare: transpose (nrow,nchan,npol) -> (npol,nrow,nchan).
    raw_vis_jublayout = np.ascontiguousarray(np.transpose(raw_data_sel, (2, 0, 1)))
    # jubik's weight = WEIGHT (broadcast over nchan) * (~flag); then vis[w==0]=0.
    nchan = raw_data_sel.shape[1]
    if raw_wspec is not None:
        raw_wgt_bc = np.transpose(raw_wspec[sel], (2, 0, 1))
    else:
        raw_wgt_bc = np.repeat(raw_weight_sel[:, None, :], nchan, axis=1)
        raw_wgt_bc = np.transpose(raw_wgt_bc, (2, 0, 1))         # (npol,nrow,nchan)
    raw_wgt_expected = raw_wgt_bc * (~np.transpose(raw_flag_sel, (2, 0, 1)))
    raw_vis_expected = raw_vis_jublayout.copy()
    raw_vis_expected[raw_wgt_expected == 0] = 0.0

    rows = []

    def report(name: str, ok: bool, detail: str) -> None:
        rows.append((name, ok, detail))

    # geometry & indices: must be byte-identical, in the SAME row order
    report("UVW identical (sign untouched)",
           np.array_equal(jub_uvw, raw_uvw),
           f"max|Δ|={_row_max_absdiff(jub_uvw, raw_uvw):.3e}, "
           f"shape {jub_uvw.shape}")
    report("ANTENNA1 identical (row order)",
           np.array_equal(jub_ant1, raw_ant1),
           f"n={jub_ant1.size}, mismatches={int(np.sum(jub_ant1 != raw_ant1))}")
    report("ANTENNA2 identical (row order)",
           np.array_equal(jub_ant2, raw_ant2),
           f"n={jub_ant2.size}, mismatches={int(np.sum(jub_ant2 != raw_ant2))}")
    report("TIME identical (row order)",
           np.array_equal(jub_time, raw_time),
           f"max|Δ|={_row_max_absdiff(jub_time, raw_time):.3e}")

    # frequency & phase center
    report("freq == CHAN_FREQ[spw]",
           np.array_equal(jub_freq, chan_freq),
           f"max|Δ|={_row_max_absdiff(jub_freq, chan_freq):.3e}, "
           f"{jub_freq.tolist()}")

    # visibilities: per-polarization, against the flag/weight-zeroed raw column
    vis_diff_raw = _row_max_absdiff(jub_vis.view(np.float64),
                                    raw_vis_expected.view(np.float64))
    report("vis == transpose(DATA), zeroed where w==0",
           np.array_equal(jub_vis, raw_vis_expected),
           f"max|Δ|={vis_diff_raw:.3e} over all pols")
    for p, lab in enumerate(pol_labels):
        d = _row_max_absdiff(jub_vis[p].view(np.float64),
                             raw_vis_expected[p].view(np.float64))
        report(f"  pol[{p}]={lab}: vis[{p}] == DATA[...,{p}]",
               np.array_equal(jub_vis[p], raw_vis_expected[p]),
               f"max|Δ|={d:.3e}")

    # weights
    report("weight == WEIGHT * (~FLAG)",
           np.array_equal(jub_wgt, raw_wgt_expected),
           f"max|Δ|={_row_max_absdiff(jub_wgt, raw_wgt_expected):.3e}")

    # --- THE conjugation test (the involution-critical one) -----------------
    # Compare jubik vis against the raw DATA and against its complex conjugate.
    raw_conj = np.conj(raw_vis_jublayout)
    d_plain = float(np.max(np.abs(jub_vis - raw_vis_jublayout)))
    d_conj = float(np.max(np.abs(jub_vis - raw_conj)))
    imag_scale = float(np.max(np.abs(raw_data_sel.imag)))
    no_conj = d_plain <= d_conj
    report("NO conjugation: |vis-DATA| <= |vis-conj(DATA)|",
           no_conj,
           f"|vis-DATA|={d_plain:.3e}  |vis-conj(DATA)|={d_conj:.3e}  "
           f"max|Im(DATA)|={imag_scale:.3e}")

    # phase center (casacore is the anchor; jubik exposes it via the FIELD
    # auxiliary table, which its `direction` property reads verbatim)
    aux_ref = np.asarray(obs.auxiliary_table("FIELD")["REFERENCE_DIR"][FIELD][0])
    aux_phase = np.asarray(obs.auxiliary_table("FIELD")["PHASE_DIR"][FIELD][0])
    report("phase center: jubik FIELD.PHASE_DIR == MS PHASE_DIR",
           np.array_equal(aux_phase, phase_dir),
           f"max|Δ|={_row_max_absdiff(aux_phase, phase_dir):.3e}, "
           f"rad {phase_dir.tolist()}")
    report("phase center: jubik FIELD.REFERENCE_DIR == MS REFERENCE_DIR",
           np.array_equal(aux_ref, reference_dir),
           f"max|Δ|={_row_max_absdiff(aux_ref, reference_dir):.3e}")

    # --- (4) npz round trip --------------------------------------------------
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".npz", dir=str(SCRATCH), delete=False
    )
    tmp.close()
    obs.save(tmp.name, compress=False)
    obs2 = obs.__class__.load(tmp.name)
    rt_ok = (
        np.array_equal(np.asarray(obs2.vis_val), jub_vis)
        and np.array_equal(np.asarray(obs2.weight_val), jub_wgt)
        and np.array_equal(np.asarray(obs2.uvw), jub_uvw)
        and np.array_equal(np.asarray(obs2.ant1), jub_ant1)
        and np.array_equal(np.asarray(obs2.ant2), jub_ant2)
        and np.array_equal(np.asarray(obs2.time), jub_time)
        and np.array_equal(np.asarray(obs2.freq), jub_freq)
        and obs2.legacy_polarization.to_str_list() == jub_pol
    )
    report("npz round trip (save->load) byte-identical",
           rt_ok,
           f"file {Path(tmp.name).name} in scratchpad")
    os.unlink(tmp.name)

    # --- findings table ------------------------------------------------------
    print("FINDINGS")
    print("-" * 74)
    width = max(len(n) for n, _, _ in rows)
    for name, ok, detail in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<{width}}  {detail}")
    print("-" * 74)

    # --- assertions: ONLY the involution-critical facts ---------------------
    assert no_conj and d_plain == 0.0, (
        "CONJUGATION detected: jubik visibilities differ from the raw DATA "
        f"column (|vis-DATA|={d_plain:.3e}, |vis-conj|={d_conj:.3e}). The "
        "import must transcribe DATA without conjugation."
    )
    assert np.array_equal(jub_ant1, raw_ant1), "ANTENNA1 row order altered"
    assert np.array_equal(jub_ant2, raw_ant2), "ANTENNA2 row order altered"
    assert np.array_equal(jub_time, raw_time), "TIME row order altered"
    assert np.array_equal(jub_uvw, raw_uvw), (
        "UVW altered (sign flip or reorder) relative to the raw MS column"
    )
    assert np.array_equal(jub_vis, raw_vis_expected), (
        "visibilities differ from transpose(DATA) with the vis[w==0]=0 rule"
    )
    assert rt_ok, "npz round trip is not byte-identical"

    print("\nVERDICT: MS import is FAITHFUL — no conjugation, no row reorder, "
          "uvw sign untouched, npz round trip exact.")


if __name__ == "__main__":
    main()
