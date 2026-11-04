"""load_hst_exposure on synthetic NICMOS _cal and WFPC2 _flt files."""

import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from jubik.color import Color
from jubik.grid import Grid
from jubik.instruments.hst.hst_data import load_hst_exposure, ramp_poisson_factor
from jubik.instruments.jwst.integration.integration import integration_factory
from jubik.instruments.jwst.integration.unit_conversion import build_unit_conversion
from jubik.instruments.jwst.likelihood.likelihood import (
    ShotNoiseGaussianLikelihoodBuilder,
)
from jubik.instruments.jwst.throughput.weights import Throughput, throughput_weights
from jubik.wcs import WcsAstropy

SHAPE = (40, 50)  # rows, columns
ROWS, COLUMNS = np.mgrid[: SHAPE[0], : SHAPE[1]]
RAMP = 100.0 * ROWS + COLUMNS + 1  # every pixel value tells its position
SR_PER_ARCSEC2 = (np.pi / 648000) ** 2
C_ANGSTROM_PER_S = 2.99792458e18
JY_PER_CGS = 1e23  # Jy per erg / s / cm^2 / Hz

NIC2_SCALE = (0.0760, 0.0754)  # arcsec, x and y
NIC2_PHOT = dict(PHOTFNU=1.5e-6, PHOTFLAM=2.0e-19, PHOTPLAM=16000.0)
WF_SCALE = (0.0996, 0.0996)
WF_PHOT = dict(PHOTFLAM=2.5e-18, PHOTPLAM=8000.0)  # WFPC2 files carry no PHOTFNU
WF_PHOTFNU = WF_PHOT["PHOTFLAM"] * WF_PHOT["PHOTPLAM"] ** 2 / C_ANGSTROM_PER_S
WF_PHOTFNU *= JY_PER_CGS
EXPTIME = 400.0
ADCGAIN = 5.4
NSAMP = 11  # reads after the zeroth
READ_TIMES = np.linspace(0.0, EXPTIME, NSAMP + 1)  # SAMPTIME, zeroth first
RAMP_TIME = READ_TIMES[-1] - READ_TIMES[1]  # calnica's TIME: from read 1

CURVE = Throughput([1.4, 1.8] * u.um, np.ones(2))
SPECTRAL = Color(np.array([[1.0, 1.5], [1.5, 2.0]]) * u.um)

# The default grid (_grid) has pixel centres at chip x = 10.75..19.75 and
# y = 8.75..15.75, so chip pixel centres inside it are rows 9..16, columns
# 11..20. The cutout uses world_corners one grid pixel out, so that it holds
# every pixel the mask keeps (up to half a grid pixel past the far edge):
# x = 10.25, 21.25 and y = 8.25, 17.25, box rows 8:19, columns 10:23, plus
# the margin of 2.
CUT = np.s_[6:21, 8:25]
INSIDE = np.zeros(SHAPE, bool)
INSIDE[9:17, 11:21] = True
INSIDE = INSIDE[CUT]


def _sci_header(scale, *, bunit, sip=True, **cards):
    h = fits.Header()
    h["BUNIT"] = bunit
    h["RADESYS"] = "ICRS"
    h["CTYPE1"], h["CTYPE2"] = (
        ("RA---TAN-SIP", "DEC--TAN-SIP") if sip else ("RA---TAN", "DEC--TAN")
    )
    h["CRPIX1"], h["CRPIX2"] = 25.5, 20.5
    h["CRVAL1"], h["CRVAL2"] = 150.0, 2.0
    h["CD1_1"], h["CD1_2"] = -scale[0] / 3600, 0.0
    h["CD2_1"], h["CD2_2"] = 0.0, scale[1] / 3600
    if sip:  # moves pixels of the default cutout by up to 0.06 px
        h["A_ORDER"] = h["B_ORDER"] = 2
        h["A_2_0"], h["B_0_2"] = 3e-4, -3e-4
    h.update(cards)
    return h


def _chip(ver, header, data, err=True):
    hdus = [fits.ImageHDU(data, header, name="SCI", ver=ver)]
    if err:
        hdus.append(fits.ImageHDU(np.full(SHAPE, 0.5), name="ERR", ver=ver))
    hdus.append(fits.ImageHDU(np.zeros(SHAPE, np.int16), name="DQ", ver=ver))
    return hdus


def _nicmos(path, *, camera=2, bunit="COUNTS/S", err=True, sip=True):
    primary = fits.PrimaryHDU()
    primary.header.update(
        INSTRUME="NICMOS",
        CAMERA=camera,
        EXPTIME=EXPTIME,
        ADCGAIN=ADCGAIN,
        OBSMODE="MULTIACCUM",
    )
    primary.header.update(NIC2_PHOT)
    header = _sci_header(NIC2_SCALE, bunit=bunit, sip=sip)
    ramp = [
        fits.ImageHDU(np.full(SHAPE, NSAMP, np.int16), name="SAMP", ver=1),
        fits.ImageHDU(np.full(SHAPE, RAMP_TIME), name="TIME", ver=1),
    ]
    fits.HDUList([primary, *_chip(1, header, RAMP, err), *ramp]).writeto(path)
    return path


def _wfpc2(path, *, bunit="COUNTS"):
    # MAST layout: SCI,1-4 first, then the DQ and ERR of each chip; the primary
    # DETECTOR is a string, the per-chip one an integer
    primary = fits.PrimaryHDU()
    primary.header.update(INSTRUME="WFPC2", DETECTOR="PC", EXPTIME=EXPTIME)
    chips = [
        # one WCS for all chips, so _grid fits every chip
        _chip(
            d, _sci_header(WF_SCALE, bunit=bunit, DETECTOR=d, **WF_PHOT), RAMP + 1e4 * d
        )
        for d in (1, 2, 3, 4)
    ]
    hdus = [primary, *(c[0] for c in chips), *(h for c in chips for h in c[1:])]
    fits.HDUList(hdus).writeto(path)
    return path


def _grid(path, center_xy=(15.25, 12.25), shape_xy=(10, 8), position_angle=0):
    """Sky grid with the chip's pixel size, centred on chip pixel `center_xy`."""
    with fits.open(path) as hdul:
        header = hdul["SCI", 1].header
        center = WCS(header, hdul).pixel_to_world(*center_xy)
    scale = (-header["CD1_1"] * 3600, header["CD2_2"] * 3600)
    fov = tuple(n * s * u.arcsec for n, s in zip(shape_xy, scale))
    spatial = WcsAstropy(center, shape_xy, fov, position_angle * u.deg)
    return Grid(spatial, SPECTRAL)


def _load(path, **kwargs):
    kwargs = dict(margin=2, dq_bits=0) | kwargs
    return load_hst_exposure(path, kwargs.pop("grid", _grid(path)), CURVE, **kwargs)


def _constant(path, name, value):
    """Store extension `name` as HST stores a constant array: PIXVALUE, no data."""
    with fits.open(path) as hdul:
        hdus = [h.copy() for h in hdul]
    i = [h.name for h in hdus].index(name)
    hdus[i] = fits.ImageHDU(name=name, ver=1)
    hdus[i].header.update(PIXVALUE=value, NPIX1=SHAPE[1], NPIX2=SHAPE[0])
    fits.HDUList(hdus).writeto(path, overwrite=True)


@pytest.mark.parametrize("subsample", [1, 2])
@pytest.mark.parametrize("bunit", ["COUNTS/S", "COUNTS"])
@pytest.mark.parametrize("instrument", ["nicmos", "wfpc2"])
def test_uniform_sky_predicts_the_native_value(tmp_path, instrument, bunit, subsample):
    if instrument == "nicmos":  # PHOTFNU in the header
        path, chip, offset = _nicmos(tmp_path / "n_cal.fits", bunit=bunit), None, 0
        scale, photfnu = NIC2_SCALE, NIC2_PHOT["PHOTFNU"]
    else:  # PHOTFLAM and PHOTPLAM only
        path, chip, offset = _wfpc2(tmp_path / "w_flt.fits", bunit=bunit), 2, 2e4
        scale, photfnu = WF_SCALE, WF_PHOTFNU
    exposure = _load(path, chip=chip, subsample=subsample)
    meta, target = exposure.meta, exposure.target_data

    fine = tuple(n * subsample for n in target.data.shape[1:])
    convert = build_unit_conversion(
        u.MJy / u.sr, (0.05 * u.arcsec) ** 2, meta.unit, meta.dvol / subsample**2
    )
    predicted = integration_factory(meta.unit, fine, subsample)(convert(np.ones(fine)))

    # WFPC2 counts are over EXPTIME; NICMOS counts become rates over TIME
    counts = bunit == "COUNTS"
    seconds = EXPTIME if counts and instrument == "wfpc2" else 1.0
    time = RAMP_TIME if counts and instrument == "nicmos" else 1.0
    # 1 MJy/sr over one pixel in PHOTFNU Jy per count/s, times EXPTIME for WFPC2
    # counts
    expected = 1e6 * scale[0] * scale[1] * SR_PER_ARCSEC2 / photfnu * seconds
    np.testing.assert_allclose(predicted, expected, rtol=1e-12)
    np.testing.assert_array_equal(target.data[0], RAMP[CUT] / time + offset)
    # WFPC2 ERR is sqrt(SCI) without gain or read noise, so no σ is given
    np.testing.assert_array_equal(
        target.std[0], 0.5 / time if instrument == "nicmos" else np.nan
    )


def test_wfpc2_mask_does_not_follow_err(tmp_path):
    path = _wfpc2(tmp_path / "w_flt.fits")
    with fits.open(path, mode="update") as hdul:  # DrizzlePac: NaN ERR where SCI < 0
        hdul["SCI", 2].data[12, 15] = -3.0
        hdul["ERR", 2].data[12, 15] = np.nan
    target = _load(path, chip=2).target_data
    assert target.mask[0][12 - CUT[0].start, 15 - CUT[1].start]
    np.testing.assert_array_equal(target.mask[0], INSIDE)
    assert np.isnan(target.std).all()


def test_products_shapes_weights_diagnostics_and_file_untouched(tmp_path):
    path = _wfpc2(tmp_path / "w_flt.fits")
    before = path.read_bytes()
    exposure = _load(path, chip=3, subsample=2, dq_bits=4)
    assert path.read_bytes() == before

    target = exposure.target_data
    assert target.data.shape == target.std.shape == target.mask.shape == (1, 15, 17)
    assert target.psf is None and target.subsample == 2
    assert len(target.subsample_centers) == 1
    assert target.subsample_centers[0].shape == (30, 34)
    np.testing.assert_array_equal(target.data[0], RAMP[CUT] + 3e4)
    np.testing.assert_array_equal(target.mask[0], INSIDE)

    meta = exposure.meta
    assert meta.unit.to(u.Jy) == pytest.approx(WF_PHOTFNU / EXPTIME, rel=1e-12)
    assert meta.dvol.to_value(u.arcsec**2) == pytest.approx(WF_SCALE[0] ** 2)
    assert meta.pixel_scale.to_value(u.arcsec) == pytest.approx(WF_SCALE[0])
    assert meta.color.to_value(u.AA) == pytest.approx(WF_PHOT["PHOTPLAM"])

    expected = throughput_weights(SPECTRAL, CURVE)
    assert exposure.weights.channels == expected.channels
    np.testing.assert_allclose(exposure.weights.weights, expected.weights)

    d = exposure.diagnostics
    assert d.file == str(path)
    assert d.extensions == (("SCI", 3), ("ERR", 3), ("DQ", 3))
    assert d.bounds == (6, 21, 8, 25)
    assert (d.bunit, d.exptime, d.dq_bits) == ("COUNTS", EXPTIME, 4)
    assert d.n_masked == INSIDE.size - INSIDE.sum()
    assert d.phot == WF_PHOT
    assert d.distortion == {"A_ORDER": 2, "B_ORDER": 2}


def test_cutout_is_clipped_to_the_chip(tmp_path):
    path = _nicmos(tmp_path / "n_cal.fits")
    with pytest.raises(ValueError, match="margin"):
        _load(path, margin=-1)
    exposure = _load(path, grid=_grid(path, center_xy=(2.25, 35.25)))
    assert exposure.diagnostics.bounds == (29, 40, 0, 12)  # one grid pixel out
    with pytest.raises(ValueError, match="misses the chip"):
        _load(path, grid=_grid(path, center_xy=(70.25, 12.25)))


def test_pixels_outside_a_rotated_grid_are_masked(tmp_path):
    path = _nicmos(tmp_path / "n_cal.fits")
    grid = _grid(path, center_xy=(25.25, 20.25), shape_xy=(10, 10), position_angle=45)
    mask = _load(path, grid=grid, margin=0).target_data.mask[0]
    centre = tuple(n // 2 for n in mask.shape)
    assert mask[centre]
    assert not (mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1])


def test_masks(tmp_path):
    path = tmp_path / "n_cal.fits"
    _nicmos(path)
    with fits.open(path, mode="update") as hdul:
        hdul["DQ"].data[10, 12] = 4  # selected bit
        hdul["DQ"].data[10, 13] = 8 | 1  # unselected bits
        hdul["SCI"].data[11, 12] = np.nan
        hdul["ERR"].data[11, 13] = 0.0
        hdul["ERR"].data[11, 14] = np.nan
    exclude = np.zeros(SHAPE, bool)
    exclude[12, 12] = True

    def cut(points):
        flags = np.zeros(SHAPE, bool)
        flags[tuple(np.transpose(points))] = True
        return flags[CUT]

    exposure = _load(path, dq_bits=4 | 2, exclude=exclude)
    not_finite = cut([(11, 12), (11, 13), (11, 14)])
    np.testing.assert_array_equal(exposure.target_data.nan_mask[0], ~not_finite)
    masked = not_finite | cut([(10, 12), (12, 12)])
    np.testing.assert_array_equal(exposure.target_data.mask[0], INSIDE & ~masked)
    assert exposure.diagnostics.n_masked == INSIDE.size - INSIDE.sum() + 5

    every_bit = _load(path, dq_bits=0xFFFF).target_data.mask[0]
    np.testing.assert_array_equal(
        every_bit, INSIDE & ~(not_finite | cut([(10, 12), (10, 13)]))
    )


@pytest.mark.parametrize("obsmode", ["ACCUM", "BRIGHTOBJ"])
@pytest.mark.parametrize("bunit, seconds", [("COUNTS/S", 1.0), ("COUNTS", 8.0)])
def test_nicmos_counts_are_rates_in_every_mode(tmp_path, obsmode, bunit, seconds):
    path = _nicmos(tmp_path / "n_cal.fits", bunit=bunit)
    time = np.full(SHAPE, 8.0)
    time[CUT][3, 3] = -0.302  # no valid sample, as calnica writes it
    _rewrite(path, dict(OBSMODE=obsmode), TIME=time)
    for constant in (False, True):
        if constant:
            _constant(path, "TIME", 8.0)
        exposure = _load(path)
        target = exposure.target_data
        mask = target.mask[0]
        np.testing.assert_array_equal(target.data[0][mask], RAMP[CUT][mask] / seconds)
        np.testing.assert_array_equal(target.std[0][mask], 0.5 / seconds)
        assert mask[3, 3] == constant
        assert exposure.noise is None
        assert exposure.meta.unit.to(u.Jy) == pytest.approx(NIC2_PHOT["PHOTFNU"])


def test_constant_extensions_stored_as_pixvalue(tmp_path):
    path = _nicmos(tmp_path / "n_cal.fits")
    _constant(path, "DQ", 0)
    np.testing.assert_array_equal(_load(path, dq_bits=-1).target_data.mask[0], INSIDE)
    _constant(path, "ERR", 0)
    with pytest.raises(ValueError, match="every pixel .* std <= 0"):
        _load(path)


def test_chip_and_camera_selection(tmp_path):
    wfpc2 = _wfpc2(tmp_path / "w_flt.fits")
    for chip in (1, 4):
        exposure = _load(wfpc2, chip=chip)
        np.testing.assert_array_equal(
            exposure.target_data.data[0], RAMP[CUT] + 1e4 * chip
        )
    for chip in (None, 5):
        with pytest.raises(ValueError, match=r"chips are \[1, 2, 3, 4\]"):
            _load(wfpc2, chip=chip)

    nicmos = _nicmos(tmp_path / "n_cal.fits", camera=2)
    assert _load(nicmos, chip=2).diagnostics.extensions[0] == ("SCI", 1)
    with pytest.raises(ValueError, match=r"chips are \[2\]"):
        _load(nicmos, chip=1)


def test_missing_err_raises(tmp_path):
    with pytest.raises(ValueError, match="no ERR extension"):
        _load(_nicmos(tmp_path / "n_cal.fits", err=False))


@pytest.mark.parametrize("subsample", [1, 2])
def test_subpixel_centres_apply_the_distortion(tmp_path, subsample):
    path = _wfpc2(tmp_path / "w_flt.fits")
    exposure = _load(path, chip=2, subsample=subsample)
    centers = exposure.target_data.subsample_centers[0]

    # fine pixel i of data pixel p sits at p + (i + 1/2) / s - 1/2
    offsets = (np.arange(subsample) + 0.5) / subsample - 0.5
    x = (np.arange(CUT[1].start, CUT[1].stop)[:, None] + offsets).ravel()
    y = (np.arange(CUT[0].start, CUT[0].stop)[:, None] + offsets).ravel()
    xx, yy = np.meshgrid(x, y)
    with fits.open(path) as hdul:
        wcs = WCS(hdul["SCI", 2].header, hdul)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    np.testing.assert_allclose(centers.ra.deg, ra, rtol=0, atol=1e-12)
    np.testing.assert_allclose(centers.dec.deg, dec, rtol=0, atol=1e-12)

    linear = SkyCoord(*wcs.wcs_pix2world(xx, yy, 0), unit="deg")
    assert centers.separation(linear).max() > 1 * u.mas


def test_header_without_distortion_raises(tmp_path):
    with pytest.raises(ValueError, match="updatewcs"):
        _load(_nicmos(tmp_path / "n_cal.fits", sip=False))


# --------------------------------------------------------------------------- #
# NICMOS shot noise
# --------------------------------------------------------------------------- #

# SPARS64 read times after the zeroth read, zeroth included (NICMOS Data
# Handbook v8, p. 90, sampinfo of n4ux23x0q)
SPARS64 = np.r_[0.0, 0.302, 0.605, 63.993 + 63.997 * np.arange(22)]
# STEP256 SAMPTIME of the _ima of n44507qgq (GO 7255, NIC1, calnica 4.4.1)
STEP256 = np.array(
    [0.0, 0.3023, 0.6047, 0.993, 1.9906, 3.9846, 7.9784, 15.972, 31.9651]
    + [63.9643, 127.9615, 255.9546, 511.9538, 767.953, 1023.9523, 1279.9515]
    + [1535.9507, 1791.9499, 2047.9492, 2303.9484, 2559.9476, 2815.9468]
    + [3071.9461, 3327.9453, 3583.9445, 3839.9437]
)


def _calnica_ramp(read_times, n):
    """SAMP and TIME of a pixel whose fit used the first n reads after the
    zeroth, as calnica writes them: TIME from read 1 to read n."""
    return n, read_times[n] - read_times[1]


def _slope_weights(t, power=0.0):
    """Slope weights of the least-squares line through reads at t, read i of n
    weighted |i - (n-1)/2|^power, from numpy's polyfit (whose w is the root)."""
    t = np.asarray(t, float)
    v = np.abs(np.arange(t.size) - (t.size - 1) / 2) ** power
    return np.array([np.polyfit(t, e, 1, w=np.sqrt(v))[0] for e in np.eye(t.size)])


# calnica's read-weight power for a ramp's S/N (Fixsen et al. 2000)
SNR_LIMITS, POWERS = [5, 10, 20, 50, 100], np.array([0.0, 0.4, 1.0, 3.0, 6.0, 10.0])


def _fixsen_power(snr):
    return POWERS[np.searchsorted(SNR_LIMITS, snr, side="right")]


def _signal_for_snr(snr, read, per_count):
    """Counts s with s / sqrt(read² + s · per_count) = snr."""
    q = snr**2 * per_count
    return (q + np.sqrt(q**2 + 4 * snr**2 * read**2)) / 2


def _rewrite(path, cards=None, **arrays):
    """Replace primary cards and chip-shaped extension arrays."""
    with fits.open(path) as hdul:
        hdus = [h.copy() for h in hdul]
    hdus[0].header.update(cards or {})
    for h in hdus[1:]:
        if h.name in arrays:
            h.data = np.asarray(arrays[h.name])
    fits.HDUList(hdus).writeto(path, overwrite=True)


@pytest.mark.parametrize("n", [2, 3, 10, 100])
def test_ramp_factor_of_equal_reads(n):
    duration = 250.0
    expected = 6 * (n**2 + 1) / (5 * n * (n + 1)) / duration
    times = np.linspace(0.0, duration, n)
    assert ramp_poisson_factor(times) == pytest.approx(expected, rel=1e-12)
    if n == 2:
        assert ramp_poisson_factor(times) * duration == pytest.approx(1.0)
    if n == 100:
        assert ramp_poisson_factor(times) * duration == pytest.approx(1.188, abs=1e-3)


@pytest.mark.parametrize("power", [0.0, 0.4, 3.0, 10.0])
def test_ramp_factor_of_an_unequal_sequence(power):
    # the slope is a sum of independent Poisson increments; at unit rate each
    # increment's variance is its duration
    w = _slope_weights(SPARS64, power)
    tail = np.cumsum(w[::-1])[::-1]  # weight of the increment ending at read j
    variance = np.sum(tail[1:] ** 2 * np.diff(SPARS64))
    assert ramp_poisson_factor(SPARS64, power) == pytest.approx(variance, rel=1e-8)


def test_nicmos_b_is_the_slope_variance_of_calnica_ramps(tmp_path):
    gain, zpscale, err = ADCGAIN, 1.01, 0.002  # e-/DN, ERR in DN/s
    n_full = STEP256.size - 1
    sci, flat = np.zeros(SHAPE), np.ones(SHAPE)
    samp = np.full(SHAPE, n_full, np.int16)
    time = np.full(SHAPE, STEP256[-1] - STEP256[1], np.float32)
    # cutout pixels: (inverse flat, reads used, S/N); (5, 6) saturates
    pixels = {
        (2, 3): (1.0, n_full, 2.0),
        (2, 4): (0.9, n_full, 7.0),
        (2, 5): (1.1, n_full, 14.0),
        (2, 6): (1.0, n_full, 30.0),
        (2, 7): (1.0, n_full, 70.0),
        (4, 5): (0.7, n_full, 300.0),
        (5, 6): (1.2, 9, 300.0),
    }
    for (i, j), (f, n, snr) in pixels.items():
        chip = (i + CUT[0].start, j + CUT[1].start)
        samp[chip], time[chip] = _calnica_ramp(STEP256, n)
        read = err / np.linalg.norm(_slope_weights(STEP256[1 : n + 1]))  # counts
        signal = _signal_for_snr(snr, read, f * zpscale / gain)
        sci[chip], flat[chip] = signal / time[chip], f
    cr = (7 + CUT[0].start, 8 + CUT[1].start)
    time[cr] -= STEP256[11] - STEP256[10]  # a cosmic ray at read 10 splits the fit
    path = _nicmos(tmp_path / "n_cal.fits")
    cards = dict(ZPSCALE=zpscale, EXPTIME=3839.944)
    _rewrite(path, cards, SCI=sci, SAMP=samp, TIME=time, ERR=np.full(SHAPE, err))
    exposure = _load(path, sample_times=STEP256, flat=flat)

    for (i, j), (f, n, snr) in pixels.items():
        t = STEP256[1 : n + 1] - STEP256[1]  # the reads fitted, from read 1
        w = _slope_weights(t, _fixsen_power(snr))
        by_hand = w @ np.minimum.outer(t, t) @ w * f * zpscale / gain
        assert exposure.noise.b[i, j] == pytest.approx(by_hand, rel=1e-5), (i, j)
    two_reads = zpscale / (gain * time[cr])
    assert exposure.noise.b[7, 8] == pytest.approx(two_reads, rel=1e-6)
    sampled = exposure.target_data.data[0].size
    diagnostics = exposure.diagnostics.noise
    assert f"on {sampled - 1} of {sampled} pixels" in diagnostics
    assert "6 of them weighted by S/N" in diagnostics
    # the unweighted full ramp is 11 % above the handbook's two-read term
    full = exposure.noise.b[2, 3] * gain * time[CUT][2, 3] / zpscale
    assert full == pytest.approx(1.11, abs=0.01)


def test_nicmos_noise_matches_monte_carlo_ramps(tmp_path):
    gain, read_noise, zpscale = 5.4, 30.0, 1.5  # e-/DN, e- per read
    n_full = SPARS64.size - 1
    # cutout pixels: (rate in SCI DN/s, inverse flat, reads used, power)
    pixels = {
        (2, 3): (0.01, 1.0, n_full, 0.0),
        (4, 5): (0.3, 0.8, n_full, 3.0),
        (6, 7): (60.0, 1.3, 9, 10.0),
    }
    sci, flat, err = np.zeros(SHAPE), np.ones(SHAPE), np.full(SHAPE, 0.5)
    samp = np.full(SHAPE, n_full, np.int16)
    time = np.full(SHAPE, SPARS64[-1] - SPARS64[1])
    for (i, j), (rate, f, n, power) in pixels.items():
        chip = (i + CUT[0].start, j + CUT[1].start)
        samp[chip], time[chip] = _calnica_ramp(SPARS64, n)
        sci[chip], flat[chip] = rate, f
        read = f * zpscale * read_noise / gain  # one read, in SCI counts
        w = _slope_weights(SPARS64[1 : n + 1], power)
        err[chip] = read * np.linalg.norm(w)  # calnica's ERR
        # the S/N calnica would see puts the pixel well inside its power's range
        signal = rate * time[chip]
        w0 = _slope_weights(SPARS64[1 : n + 1])
        snr = signal / np.hypot(
            err[chip] / np.linalg.norm(w0), np.sqrt(signal * f * zpscale / gain)
        )
        assert _fixsen_power(snr) == power, snr
    path = _nicmos(tmp_path / "n_cal.fits")
    cards = dict(ADCGAIN=gain, ZPSCALE=zpscale, EXPTIME=SPARS64[-1])
    _rewrite(path, cards, SCI=sci, SAMP=samp, TIME=time, ERR=err)
    noise = _load(path, sample_times=SPARS64, flat=flat).noise

    rng = np.random.default_rng(172)
    for (i, j), (rate, f, n, power) in pixels.items():
        t = SPARS64[: n + 1]  # charge accumulates from the zeroth read
        electrons_per_s = rate / (f * zpscale) * gain  # SCI = inverse flat * raw
        counts = np.cumsum(rng.poisson(electrons_per_s * np.diff(t), (40_000, n)), 1)
        reads = counts + rng.normal(0, read_noise, (40_000, n))  # reads 1..n
        sci = (reads / gain @ _slope_weights(t[1:], power)) * f * zpscale
        predicted = noise.a[i, j] + noise.b[i, j] * rate
        # the variance of 40000 draws is good to 0.7 %
        assert sci.var() == pytest.approx(predicted, rel=0.03), (i, j)
        assert sci.mean() == pytest.approx(rate, rel=0.02, abs=0.01)
    assert noise.b[6, 7] > 2 * noise.b[4, 5]  # fewer reads, larger factor


def test_nicmos_noise_defaults_and_units(tmp_path):
    path = _nicmos(tmp_path / "n_cal.fits")
    time = np.full(SHAPE, RAMP_TIME)
    time[CUT][1, 1] = 123.0  # no read at this time: the handbook's term
    _rewrite(path, TIME=time, SCI=np.zeros(SHAPE))  # S/N < 5: unweighted
    dark = np.full(SHAPE, 0.25)
    rate = _load(path, sample_times=READ_TIMES, dark=dark)
    np.testing.assert_allclose(rate.noise.a, 0.25)  # ERR**2
    np.testing.assert_allclose(rate.noise.sky, 0.25)
    n = NSAMP
    expected = 6 * (n**2 + 1) / (5 * n * (n + 1)) / (ADCGAIN * RAMP_TIME)
    assert rate.noise.b[0, 0] == pytest.approx(expected, rel=1e-12)
    assert rate.noise.b[1, 1] == pytest.approx(1 / (ADCGAIN * 123.0), rel=1e-12)
    assert f"on {INSIDE.size - 1} of {INSIDE.size} pixels" in rate.diagnostics.noise
    handbook = _load(path)  # no sample times
    assert handbook.noise.b[0, 0] == pytest.approx(1 / (ADCGAIN * RAMP_TIME))
    assert handbook.noise.sky == 0.0

    assert _load(_wfpc2(tmp_path / "w_flt.fits"), chip=3).noise is None


def test_nicmos_counts_and_rates_give_the_same_likelihood(tmp_path):
    # calnica's UNITCORR: COUNTS/S = COUNTS / TIME, with TIME per pixel
    rng = np.random.default_rng(172)
    samp = rng.integers(2, NSAMP + 1, SHAPE)  # shortened ramps, e.g. saturated
    time = READ_TIMES[samp] - READ_TIMES[1]
    time[CUT][3, 4] -= READ_TIMES[2] - READ_TIMES[1]  # a cosmic-ray split ramp
    time[CUT][5, 6] = 0.0  # no valid sample
    sci = rng.uniform(0.01, 300.0, SHAPE)  # S/N across calnica's weight powers
    err, dark = rng.uniform(0.2, 0.5, SHAPE), rng.uniform(0.0, 0.1, SHAPE)
    flat = rng.uniform(0.8, 1.2, SHAPE)
    loaded = []
    for bunit, seconds in (("COUNTS/S", 1.0), ("COUNTS", time)):
        path = _nicmos(tmp_path / f"{len(loaded)}_cal.fits", bunit=bunit)
        _rewrite(path, SCI=sci * seconds, ERR=err * seconds, SAMP=samp, TIME=time)
        loaded.append(
            _load(path, sample_times=READ_TIMES, flat=flat, dark=dark * seconds)
        )

    # MJy/sr giving about 0.01 to 300 counts/s per pixel, like SCI
    field = {"k": 10 ** rng.uniform(-0.5, 3.8, loaded[0].target_data.data.shape[1:])}
    results = []
    for exposure in loaded:
        meta, target = exposure.meta, exposure.target_data
        convert = build_unit_conversion(
            u.MJy / u.sr, (0.05 * u.arcsec) ** 2, meta.unit, meta.dvol
        )
        integrate = integration_factory(meta.unit, field["k"].shape, 1)
        response = jft.Model(
            lambda x: integrate(convert(x["k"]))[target.mask[0]],
            domain={"k": jft.ShapeWithDtype(field["k"].shape, float)},
        )
        builder = ShotNoiseGaussianLikelihoodBuilder(
            response, target.data, exposure.noise, target.mask
        )
        prediction, inverse_std = builder.model()(field)
        results.append((target.mask, prediction, inverse_std, builder.build()(field)))
    rate, counts = results
    np.testing.assert_array_equal(counts[0], rate[0])
    assert not rate[0][0, 5, 6] and rate[0][0, 3, 4]
    for r, c in zip(rate[1:], counts[1:]):
        np.testing.assert_allclose(c, r, rtol=1e-9)
    assert loaded[1].meta.unit.to(u.Jy) == loaded[0].meta.unit.to(u.Jy)
    np.testing.assert_allclose(
        loaded[1].target_data.data[0][rate[0][0]], sci[CUT][rate[0][0]], rtol=1e-12
    )


def test_nicmos_noise_rejects_bad_inputs(tmp_path):
    path = _nicmos(tmp_path / "n_cal.fits")
    with pytest.raises(ValueError, match="flat has shape"):
        _load(path, flat=np.ones((3, 3)))
    with pytest.raises(ValueError, match="sample_times"):
        _load(path, sample_times=[np.nan])
    with pytest.raises(ValueError, match="not EXPTIME"):  # planned, not real
        _load(path, sample_times=READ_TIMES + np.r_[0, np.full(NSAMP, 0.02)])
    _rewrite(path, dict(ADCGAIN=0))
    with pytest.raises(ValueError, match="ADCGAIN"):
        _load(path)


def test_nicmos_edge_pixels_and_modes(tmp_path):
    path = _nicmos(tmp_path / "n_cal.fits")
    samp, time = np.full(SHAPE, NSAMP), np.full(SHAPE, RAMP_TIME)
    samp[CUT][2, 2] = 0  # no read used, though TIME is the full ramp's
    samp[CUT][5, 5] = NSAMP + 1  # more reads than the sequence has
    time[CUT][3, 3] = 0.0  # no valid sample
    _rewrite(path, SAMP=samp, TIME=time)
    exposure = _load(path, sample_times=READ_TIMES)
    for pixel in ((2, 2), (5, 5)):
        assert exposure.noise.b[pixel] == pytest.approx(1 / (ADCGAIN * RAMP_TIME))
    assert not exposure.target_data.mask[0, 3, 3]
    dark = np.zeros(SHAPE)
    dark[CUT][4, 4] = np.nan
    without_zeroth = _load(path, sample_times=READ_TIMES[1:], dark=dark)
    np.testing.assert_array_equal(without_zeroth.noise.b, exposure.noise.b)
    assert not without_zeroth.target_data.mask[0, 4, 4]
    _rewrite(path, dict(OBSMODE="ACCUM"))  # ERR holds the Poisson term
    assert _load(path).noise is None
    with pytest.raises(ValueError, match="MULTIACCUM only"):
        _load(path, flat=np.ones(SHAPE))


def test_nicmos_normalized_residuals_through_the_response(tmp_path):
    read_noise = 30.0  # e- per read
    t = READ_TIMES[1:]  # calnica fits the reads after the zeroth
    weights = np.array([_slope_weights(t, p) for p in POWERS])
    rng = np.random.default_rng(172)
    flat = rng.uniform(0.8, 1.2, SHAPE)  # inverse flat
    err = flat * read_noise / ADCGAIN * np.linalg.norm(weights[0])  # calnica's ERR
    path = _nicmos(tmp_path / "n_cal.fits")
    _rewrite(path, ERR=err)
    exposure = _load(path, sample_times=READ_TIMES, flat=flat)
    meta, shape = exposure.meta, exposure.target_data.data.shape[1:]
    convert = build_unit_conversion(
        u.MJy / u.sr, (0.05 * u.arcsec) ** 2, meta.unit, meta.dvol
    )
    integrate = integration_factory(meta.unit, shape, 1)
    per_unit_sky = np.mean(integrate(convert(np.ones(shape))))  # DN/s
    field = np.full(shape, 0.05 / per_unit_sky)  # read-noise dominated
    field[:, shape[1] // 2 :] = 50.0 / per_unit_sky  # shot-noise dominated
    truth = np.zeros(SHAPE)
    truth[CUT] = integrate(convert(field))
    _rewrite(path, SCI=truth)  # SCI sets calnica's weights
    exposure = _load(path, sample_times=READ_TIMES, flat=flat)
    target, mask = exposure.target_data, exposure.target_data.mask[0]
    response = jft.Model(
        lambda x: integrate(convert(x["k"]))[mask],
        domain={"k": jft.ShapeWithDtype(shape, float)},
    )
    builder = ShotNoiseGaussianLikelihoodBuilder(
        response, target.data, exposure.noise, target.mask
    )
    model, inverse_std = (np.asarray(v) for v in builder.model()({"k": field}))

    f = flat[CUT][mask]
    rate = np.broadcast_to(model / f * ADCGAIN, (300, model.size))  # e-/s
    counts = np.cumsum(rng.poisson(rate[..., None] * np.diff(READ_TIMES)), -1)
    reads = counts + rng.normal(0, read_noise, counts.shape)  # reads 1..NSAMP
    y = reads / ADCGAIN * f[:, None]  # flat-fielded DN, as in the _ima
    # calnica picks the power per ramp from its S/N, (y_n - y_1) / σ(y_n)
    sigma = np.sqrt(
        (f * read_noise / ADCGAIN)[:, None] ** 2
        + np.maximum(y, 0) * f[:, None] / ADCGAIN
    )
    snr = (y[..., -1] - y[..., 0]) / sigma[..., -1]
    power = np.searchsorted(SNR_LIMITS, snr, side="right")
    data = np.take_along_axis(y @ weights.T, power[..., None], -1)[..., 0]
    for pixels in (model < 1, model > 10):
        assert pixels.sum() > 30
        normalized = (data[:, pixels] - model[pixels]) * inverse_std[pixels]
        assert normalized.mean() == pytest.approx(0, abs=0.03)
        assert normalized.var() == pytest.approx(1, abs=0.05)
    assert np.mean(power[:, model > 10] == 5) > 0.99  # bright ramps: power 10
    bright = model > 10
    err_only = (data[:, bright] - model[bright]) / err[CUT][mask][bright]
    assert err_only.var() > 10
