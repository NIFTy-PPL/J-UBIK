# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

"""Static detector pattern: synthetic dithers, an injected pattern, against truth."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from jubik.instruments.jwst.data.static_pattern import subtract_static_pattern

# full-width rows: four 512-column amplifiers
NY, NX, BLOCK, MIN_COUNT = 96, 2048, 32, 5
SCALE = 0.063  # arcsec per pixel
TARGET = SkyCoord(294.6 * u.deg, 66.8 * u.deg)
RADIUS = 15 * SCALE * u.arcsec
TARGET_SKY = (70, 1000)  # the target's pixel in the sky frame
GROUPS = [(0, 0), (15, 60), (30, 120)]  # pointing of each group of 7, in pixels
DITHERS = [(0, 0), (1, 2), (2, 1), (3, 3), (0, 3), (2, 0), (1, 1)]
PATTERN_AMP, NOISE = 1.2, 1.0
ALWAYS_BAD = (np.array([10, 50, 80, 77]), np.array([400, 1200, 1700, 950]))  # y, x
LOW_X = slice(4, 132)  # DO_NOT_USE in 70% of the exposures: few others see sky there


def make_set(folder, noise_seed=0):
    """21 dithers of one sky, a detector-fixed pattern; exposure 0 has no ERR_PIPE."""
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    pattern = PATTERN_AMP * rng.normal(size=(NY, NX))
    sy, sx = NY + 40, NX + 130
    yy, xx = np.mgrid[:sy, :sx]
    r = np.hypot(yy - TARGET_SKY[0], xx - TARGET_SKY[1])
    # a core, and an arc-like ring below the 3-sigma clip: only the target disk masks it
    sky = 100 * np.exp(-(r**2) / 8) + 2 * np.exp(-((r - 11) ** 2) / 2.88)
    for y, x, peak in zip(
        rng.uniform(0, sy, 60), rng.uniform(0, sx, 60), rng.uniform(20, 200, 60)
    ):
        sky += peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4.5)
    dy_, dx_ = np.mgrid[:NY, :NX]
    paths = []
    for e in range(21):
        (gy, gx), (dy, dx) = GROUPS[e // 7], DITHERS[e % 7]
        oy, ox = gy + dy, gx + dx
        draw = np.random.default_rng((1, e))
        noise = np.random.default_rng((2, e, noise_seed if e == 0 else 0)).normal(
            0, NOISE, (NY, NX)
        )
        stripes = np.repeat(draw.normal(0, 0.5, (NY, 4)), NX // 4, axis=1)
        sci = (
            sky[oy : oy + NY, ox : ox + NX]
            + 30
            + 2 * dx_ / NX
            + dy_ / NY
            + stripes
            + pattern
            + noise
        )
        dq = np.zeros((NY, NX), np.uint32)
        dq[:, LOW_X][draw.random((NY, LOW_X.stop - LOW_X.start)) < 0.7] = 1
        dq[ALWAYS_BAD] = 1
        sci[ALWAYS_BAD] = np.nan

        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [TARGET.ra.deg, TARGET.dec.deg]
        w.wcs.crpix = [TARGET_SKY[1] - ox + 1, TARGET_SKY[0] - oy + 1]
        w.wcs.cdelt = [-SCALE / 3600, SCALE / 3600]
        primary = fits.PrimaryHDU()
        primary.header.update(
            DETECTOR="NRCBLONG",
            FILTER="F277W",
            PUPIL="CLEAR",
            R_DARK="crds://jwst_nircam_dark_0618.fits",
            SUBARRAY="FULL",
        )
        ones = np.ones((NY, NX), np.float32)
        hdus = [
            primary,
            fits.ImageHDU(sci.astype(np.float32), w.to_header(), name="SCI"),
        ]
        if e == 0:  # a plain cal file: ERR is the pipeline's
            hdus += [fits.ImageHDU(ones, name="ERR"), fits.ImageHDU(dq, name="DQ")]
        else:  # a cleaned file: ERR already carries a per-exposure term
            hdus += [fits.ImageHDU(2 * ones, name="ERR"), fits.ImageHDU(dq, name="DQ")]
            hdus.append(fits.ImageHDU(ones, name="ERR_PIPE"))
        paths.append(folder / f"exp{e:02d}_cal.fits")
        fits.HDUList(hdus).writeto(paths[-1])
    return paths, pattern


def read(path, *names):
    with fits.open(path) as hdul:
        return [hdul[n].data.astype(np.float64) for n in names]


def out_of(path):
    return path.with_name(f"{path.stem}_staticpattern.fits")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("set")
    paths, pattern = make_set(folder)
    rows = subtract_static_pattern(
        paths, TARGET, RADIUS, folder / "pattern.fits", min_count=MIN_COUNT, block=BLOCK
    )
    return folder, paths, pattern, rows


def test_pattern_matches_truth_within_its_noise(run):
    folder, _, pattern, _ = run
    p, count = read(folder / "pattern.fits", "PATTERN", "COUNT")
    m = count >= 18
    # a median of n Gaussian values
    expected = NOISE * np.sqrt(np.pi / 2 * np.mean(1 / count[m]))
    assert m.sum() > 10000
    assert 0.9 < np.std((p - pattern)[m]) / expected < 1.15
    assert np.corrcoef(p[m], pattern[m])[0, 1] > 0.9
    assert np.all(p[count < MIN_COUNT] == 0) and np.any(count[4:-4, LOW_X] < MIN_COUNT)
    # measured from exactly MIN_COUNT (a residual can be exactly 0 after destriping)
    assert np.mean(p[count == MIN_COUNT] != 0) > 0.9


def test_target_pixels_corrected_from_the_other_groups(run):
    _, paths, pattern, _ = run
    (sci_in,) = read(paths[0], "SCI")
    (sci_out,) = read(out_of(paths[0]), "SCI")
    ty, tx = TARGET_SKY  # exposure 0 has no pointing offset
    yy, xx = np.ogrid[:NY, :NX]
    disk = np.hypot(yy - ty, xx - tx) < 6
    assert np.all(sci_out[disk] != sci_in[disk])
    err = pattern + sci_out - sci_in
    assert np.std(err[disk]) < 0.5 < np.std(pattern[disk])  # 14 other exposures: ~0.34


def test_target_ring_leaves_no_ghost_in_the_other_groups(run):
    _, paths, pattern, _ = run
    # where group 0 sees the target's ring; group 1 (exposures 7-13) sees sky there
    ty, tx = TARGET_SKY
    yy, xx = np.ogrid[:NY, :NX]
    ring = np.abs(np.hypot(yy - ty, xx - tx) - 11) < 2
    for path in paths[7:14]:
        (sci_in,) = read(path, "SCI")
        (sci_out,) = read(out_of(path), "SCI")
        # ~0.03 by noise alone
        assert abs(np.mean((pattern + sci_out - sci_in)[ring])) < 0.1


def test_sigma_p_matches_the_corrected_scatter_at_high_and_low_count(run):
    _, paths, pattern, _ = run
    errs, sigmas = [], []
    for path in paths:
        (sci_in,) = read(path, "SCI")
        sci_out, err, pipe = read(out_of(path), "SCI", "ERR", "ERR_PIPE")
        corrected = np.isfinite(sci_in) & (sci_out != sci_in)
        errs.append((pattern + sci_out - sci_in)[corrected])
        sigmas.append(np.sqrt(err**2 - pipe**2)[corrected])
    errs, sigmas = np.concatenate(errs), np.concatenate(sigmas)
    # sigma_P scales as 1/sqrt(n); n = 20 is the most others
    at_one = sigmas.min() * np.sqrt(20)
    for sel in (sigmas <= at_one / np.sqrt(17.5), sigmas >= at_one / np.sqrt(8.5)):
        assert sel.sum() > 5000
        assert 0.85 < np.std(errs[sel]) / np.sqrt(np.mean(sigmas[sel] ** 2)) < 1.15


def test_uncorrected_pixels_keep_sci_and_the_per_exposure_term(run):
    _, paths, _, rows = run
    dqs = [read(p, "DQ")[0].astype(int) & 1 for p in paths]
    # fewer than MIN_COUNT exposures can see sky there at all
    few = np.sum([dq == 0 for dq in dqs], axis=0) < MIN_COUNT
    few &= np.isfinite(read(paths[0], "SCI")[0])
    assert few[:, LOW_X].mean() > 0.1
    for path, row, dq in zip(paths, rows, dqs):
        (sci_in,) = read(path, "SCI")
        sci_out, err, pipe = read(out_of(path), "SCI", "ERR", "ERR_PIPE")
        kept = few | ((dq > 0) & np.isfinite(sci_in))
        np.testing.assert_array_equal(sci_out[kept], sci_in[kept])
        np.testing.assert_allclose(
            err[kept], np.sqrt(pipe[kept] ** 2 + row["sigma_x_in"] ** 2), rtol=1e-6
        )
        # the pattern (1.2) is kept there, ~0.3 is left where corrected
        assert row["sigma_x_in"] > 1.0 > 3 * row["sigma_p_med"]


def test_do_not_use_and_nan_pixels_are_untouched(run):
    folder, paths, _, _ = run
    p, count = read(folder / "pattern.fits", "PATTERN", "COUNT")
    assert np.all(count[ALWAYS_BAD] == 0) and np.all(p[ALWAYS_BAD] == 0)
    for path in paths:
        sci_in, dq = read(path, "SCI", "DQ")
        (sci_out,) = read(out_of(path), "SCI")
        bad = (dq.astype(int) & 1) > 0
        np.testing.assert_array_equal(sci_out[bad], sci_in[bad])  # NaN stays NaN
    (err_in,) = read(paths[0], "ERR")
    (pipe_out,) = read(out_of(paths[0]), "ERR_PIPE")
    # a plain cal file's ERR is kept as ERR_PIPE
    np.testing.assert_array_equal(pipe_out, err_in)


def test_leave_one_out_ignores_the_exposures_own_values(tmp_path):
    runs = []
    for seed in (0, 1):  # only exposure 0's noise differs
        paths, _ = make_set(tmp_path / str(seed), noise_seed=seed)
        subtract_static_pattern(
            paths, TARGET, RADIUS, tmp_path / f"p{seed}.fits", block=BLOCK
        )
        runs.append([read(p, "SCI")[0] - read(out_of(p), "SCI")[0] for p in paths[:2]])
    # exposure 0's correction comes from the others only (float32 rounding aside) ...
    np.testing.assert_allclose(runs[0][0], runs[1][0], rtol=0, atol=1e-4)
    # ... while exposure 1's sees exposure 0
    assert np.nanmax(np.abs(runs[0][1] - runs[1][1])) > 0.01


def test_refuses_mixed_sets_and_never_overwrites(tmp_path):
    paths, _ = make_set(tmp_path)
    for key, value in (
        ("DETECTOR", "NRCALONG"),
        ("FILTER", "F444W"),
        ("PUPIL", "F470N"),
        ("R_DARK", "crds://jwst_nircam_dark_0001.fits"),
        ("SUBARRAY", "SUB160"),
    ):
        old = fits.getval(paths[3], key)
        fits.setval(paths[3], key, value=value)
        with pytest.raises(
            ValueError, match=key if key != "SUBARRAY" else "full-frame"
        ):
            subtract_static_pattern(
                paths, TARGET, RADIUS, tmp_path / "p.fits", block=BLOCK
            )
        fits.setval(paths[3], key, value=old)
    with pytest.raises(ValueError, match="min_count"):  # P of no others would be NaN
        subtract_static_pattern(paths, TARGET, RADIUS, tmp_path / "p.fits", min_count=0)
    with pytest.raises(ValueError, match="min_count"):  # no exposure could be corrected
        subtract_static_pattern(
            paths[:5], TARGET, RADIUS, tmp_path / "p.fits", block=BLOCK
        )
    out_of(paths[5]).write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        subtract_static_pattern(paths, TARGET, RADIUS, tmp_path / "p.fits", block=BLOCK)
    assert out_of(paths[5]).read_bytes() == b"keep"
    assert not (tmp_path / "p.fits").exists()
