"""load_hst_exposure on synthetic NICMOS _cal and WFPC2 _flt files."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from jubik.color import Color
from jubik.instruments.hst.hst_data import load_hst_exposure
from jubik.instruments.jwst.throughput.weights import Throughput, throughput_weights

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
EXPTIME = 400.0

CURVE = Throughput([1.4, 1.8] * u.um, np.ones(2))
SPECTRAL = Color(np.array([[1.0, 1.5], [1.5, 2.0]]) * u.um)


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
    if sip:
        h["A_ORDER"] = h["B_ORDER"] = 2
        h["A_2_0"], h["B_0_2"] = 1e-3, -1e-3
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
    primary.header.update(INSTRUME="NICMOS", CAMERA=camera, EXPTIME=EXPTIME)
    primary.header.update(NIC2_PHOT)
    header = _sci_header(NIC2_SCALE, bunit=bunit, sip=sip)
    fits.HDUList([primary, *_chip(1, header, RAMP, err)]).writeto(path)
    return path


def _wfpc2(path, *, bunit="COUNTS"):
    # MAST layout: SCI,1-4 first, then the DQ and ERR of each chip; the primary
    # DETECTOR is a string, the per-chip one an integer
    primary = fits.PrimaryHDU()
    primary.header.update(INSTRUME="WFPC2", DETECTOR="PC", EXPTIME=EXPTIME)
    chips = [
        # one WCS for all chips, so _corners fits every chip
        _chip(
            d, _sci_header(WF_SCALE, bunit=bunit, DETECTOR=d, **WF_PHOT), RAMP + 1e4 * d
        )
        for d in (1, 2, 3, 4)
    ]
    hdus = [primary, *(c[0] for c in chips), *(h for c in chips for h in c[1:])]
    fits.HDUList(hdus).writeto(path)
    return path


def _corners(path, x=(10.25, 20.75), y=(8.25, 15.75)):
    """Sky corners of a pixel box, from the file's own WCS."""
    with fits.open(path) as hdul:
        wcs = WCS(hdul["SCI", 1].header, hdul)
    xx, yy = np.meshgrid(x, y)
    return wcs.pixel_to_world(xx.ravel(), yy.ravel())


def _load(path, **kwargs):
    kwargs = dict(margin=2, dq_bits=0) | kwargs
    return load_hst_exposure(
        path, kwargs.pop("corners", _corners(path)), CURVE, SPECTRAL, **kwargs
    )


def _per_count_rate(fnu_jy, scale):
    """Hand-computed MJy/sr per count/s of a pixel of `scale` arcsec."""
    return fnu_jy / (scale[0] * scale[1] * SR_PER_ARCSEC2) * 1e-6


# box from _corners: rows 8..16, columns 10..21, plus the margin of 2
CUT = np.s_[6:19, 8:24]


@pytest.mark.parametrize("bunit, per_second", [("COUNTS/S", 1), ("COUNTS", EXPTIME)])
def test_nicmos_units_from_photfnu(tmp_path, bunit, per_second):
    exposure = _load(_nicmos(tmp_path / "n_cal.fits", bunit=bunit))
    factor = _per_count_rate(NIC2_PHOT["PHOTFNU"], NIC2_SCALE) / per_second

    target = exposure.target_data
    np.testing.assert_allclose(target.data[0], RAMP[CUT] * factor, rtol=1e-12)
    np.testing.assert_allclose(target.std[0], 0.5 * factor, rtol=1e-12)
    assert exposure.diagnostics.to_mjy_sr == pytest.approx(factor, rel=1e-12)
    assert exposure.meta.unit == u.MJy / u.sr
    assert exposure.meta.dvol.to_value(u.arcsec**2) == pytest.approx(
        NIC2_SCALE[0] * NIC2_SCALE[1], rel=1e-12
    )
    assert exposure.meta.color.to_value(u.AA) == pytest.approx(16000.0)


@pytest.mark.parametrize("bunit, per_second", [("COUNTS/S", 1), ("COUNTS", EXPTIME)])
def test_wfpc2_units_from_photflam_and_photplam(tmp_path, bunit, per_second):
    exposure = _load(_wfpc2(tmp_path / "w_flt.fits", bunit=bunit), chip=2)
    fnu = WF_PHOT["PHOTFLAM"] * WF_PHOT["PHOTPLAM"] ** 2 / C_ANGSTROM_PER_S * JY_PER_CGS
    factor = _per_count_rate(fnu, WF_SCALE) / per_second

    np.testing.assert_allclose(
        exposure.target_data.data[0], (RAMP[CUT] + 2e4) * factor, rtol=1e-12
    )
    np.testing.assert_allclose(exposure.target_data.std[0], 0.5 * factor, rtol=1e-12)


def test_products_shapes_weights_diagnostics_and_file_untouched(tmp_path):
    path = _wfpc2(tmp_path / "w_flt.fits")
    before = path.read_bytes()
    exposure = _load(path, chip=3, subsample=2, dq_bits=4)
    assert path.read_bytes() == before

    target = exposure.target_data
    assert target.data.shape == target.std.shape == target.mask.shape == (1, 13, 16)
    assert target.psf is None and target.subsample == 2
    assert len(target.subsample_centers) == 1
    assert target.subsample_centers[0].shape == (26, 32)
    np.testing.assert_allclose(
        target.data[0] / exposure.diagnostics.to_mjy_sr, RAMP[CUT] + 3e4
    )

    expected = throughput_weights(SPECTRAL, CURVE)
    assert exposure.weights.channels == expected.channels
    np.testing.assert_allclose(exposure.weights.weights, expected.weights)

    d = exposure.diagnostics
    assert d.file == str(path)
    assert d.extensions == (("SCI", 3), ("ERR", 3), ("DQ", 3))
    assert d.bounds == (6, 19, 8, 24)
    assert (d.bunit, d.exptime, d.dq_bits, d.n_masked) == ("COUNTS", EXPTIME, 4, 0)
    assert d.phot == WF_PHOT
    assert d.distortion == {"A_ORDER": 2, "B_ORDER": 2}


def test_cutout_is_clipped_to_the_chip(tmp_path):
    path = _nicmos(tmp_path / "n_cal.fits")
    exposure = _load(path, corners=_corners(path, x=(0.25, 5.75), y=(30.25, 38.75)))
    assert exposure.diagnostics.bounds == (28, 40, 0, 9)
    with pytest.raises(ValueError, match="misses the chip"):
        _load(path, corners=_corners(path, x=(60.25, 70.75)))


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

    exposure = _load(path, dq_bits=4 | 2, exclude=exclude)

    def cut(points):
        flags = np.zeros(SHAPE, bool)
        flags[tuple(np.transpose(points))] = True
        return flags[CUT]

    not_finite = cut([(11, 12), (11, 13), (11, 14)])
    np.testing.assert_array_equal(exposure.target_data.nan_mask[0], ~not_finite)
    masked = not_finite | cut([(10, 12), (12, 12)])
    np.testing.assert_array_equal(exposure.target_data.mask[0], ~masked)
    assert exposure.diagnostics.n_masked == 5


def test_chip_and_camera_selection(tmp_path):
    wfpc2 = _wfpc2(tmp_path / "w_flt.fits")
    for chip in (1, 4):
        exposure = _load(wfpc2, chip=chip)
        np.testing.assert_allclose(
            exposure.target_data.data[0] / exposure.diagnostics.to_mjy_sr,
            RAMP[CUT] + 1e4 * chip,
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
    x = (np.arange(8, 24)[:, None] + offsets).ravel()
    y = (np.arange(6, 19)[:, None] + offsets).ravel()
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
