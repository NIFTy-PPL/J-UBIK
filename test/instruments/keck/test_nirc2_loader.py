"""load_nirc2_exposure: units, σ, masks, distortion and the products the JWST
target response takes. Every fixture, distortion maps included, is synthetic
and written by the tests."""

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from jubik.instruments.jwst.jwst_response import (
    build_jwst_response,
    build_sky_to_subsampled_data,
)
from jubik.instruments.jwst.parse.jwst_response import SkyMetaInformation
from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
    LinearConfig,
)
from jubik.instruments.jwst.throughput.weights import Throughput
from jubik.instruments.keck.nirc2_distortion import (
    DistortionSolution,
    select_distortion,
    service2016,
    yelda2010,
)
from jubik.instruments.keck.nirc2_loader import (
    DetectorNoise,
    FitsImage,
    Nirc2Calibration,
    load_nirc2_exposure,
)
from jubik.wcs.wcs_astropy import WcsAstropy

NY, NX = 12, 16
SCALE_MAS = 10.0
CENTER = SkyCoord(150 * u.deg, 30 * u.deg)
HEADER = {
    "CAMNAME": "narrow",
    "BUNIT": "DN",
    "ITIME": 2.0,
    "COADDS": 5,
    "DATE-OBS": "2010-06-01",
    "ROTMODE": "position angle",
    "ROTPOSN": 30.7,
    "INSTANGL": 0.7,
    "CRVAL1": 150.0,
    "CRVAL2": 30.0,
    "CRPIX1": 8.0,  # raw 0-based x = 7
    "CRPIX2": 6.0,  # raw 0-based y = 5
    "FILTER": "Kp + clear",
}
CAL = Nirc2Calibration(1e-6 * u.Jy / (u.DN / u.s), "airmass 1.0, no extinction")
BAND = Throughput([2.0, 2.4] * u.um, [1.0, 1.0])
DELTA = np.pad([[1.0]], 1)
PIXEL_SR = (SCALE_MAS * 1e-3 * np.pi / 648000) ** 2


def _per_dn(itime=2.0, coadds=5):
    """Hand-computed MJy/sr of 1 DN: 1e-6 Jy per DN/s over a 10 mas pixel."""
    return 1e-6 / (itime * coadds) / PIXEL_SR / 1e6


def _solution(dx=0.0, dy=0.0, beta=0.25, name="synthetic pre", shape=(NY, NX)):
    return DistortionSolution(
        name,
        np.broadcast_to(dx, shape),
        np.broadcast_to(dy, shape),
        SCALE_MAS * u.mas,
        beta * u.deg,
    )


def _write(path, data, header=None):
    fits.PrimaryHDU(np.asarray(data, float), fits.Header(header or {})).writeto(path)
    return FitsImage(path)


def _load(tmp_path, data=100.0, header=HEADER, **kwargs):
    folder = tmp_path / f"load{len(list(tmp_path.glob('load*')))}"
    folder.mkdir()
    science = _write(folder / "sci.fits", np.broadcast_to(data, (NY, NX)), header)
    if "detector_noise" not in kwargs:
        kwargs.setdefault("sigma", _write(folder / "sig.fits", np.full((NY, NX), 4.0)))
    kwargs.setdefault("pre_2015", _solution())
    return load_nirc2_exposure(
        science, calibration=CAL, throughput=BAND, psf=DELTA, **kwargs
    )


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


def test_dn_itime_coadds_scaling(tmp_path):
    exposure = _load(tmp_path)
    target, meta = exposure.target_data, exposure.meta
    # 100 DN / (2 s * 5) = 10 DN/s -> 1e-5 Jy in one (10 mas)^2 pixel
    np.testing.assert_allclose(target.data[0], 100 * _per_dn(), rtol=1e-12)
    np.testing.assert_allclose(target.std[0], 4 * _per_dn(), rtol=1e-12)
    np.testing.assert_allclose(target.data / target.std, 25.0)
    assert meta.unit == u.MJy / u.sr
    assert u.isclose(meta.pixel_scale, SCALE_MAS * u.mas)
    assert u.isclose(meta.dvol, (SCALE_MAS * u.mas) ** 2)
    assert exposure.diagnostics["mjy_sr_per_dn"] == pytest.approx(_per_dn())


def test_overrides_replace_header(tmp_path):
    header = {k: v for k, v in HEADER.items() if k != "BUNIT"}
    exposure = _load(tmp_path, header=header, bunit="DN", itime=4.0, coadds=1)
    np.testing.assert_allclose(
        exposure.target_data.data[0], 100 * _per_dn(4.0, 1), rtol=1e-12
    )
    np.testing.assert_allclose(
        exposure.target_data.std[0], 4 * _per_dn(4.0, 1), rtol=1e-12
    )


def test_missing_keyword_or_wrong_unit_raises(tmp_path):
    with pytest.raises(KeyError, match="ITIME"):
        _load(tmp_path, header={k: v for k, v in HEADER.items() if k != "ITIME"})
    with pytest.raises(ValueError, match="unit"):
        _load(tmp_path, bunit="DN/s")


def test_wrong_camera_raises(tmp_path):
    with pytest.raises(ValueError, match="narrow"):
        _load(tmp_path, header=HEADER | {"CAMNAME": "wide"})


def test_calibration_needs_unit_and_atmosphere():
    with pytest.raises(u.UnitConversionError):
        Nirc2Calibration(1e-6 * u.Jy, "airmass 1")
    with pytest.raises(ValueError, match="atmospheric"):
        Nirc2Calibration(1e-6 * u.Jy / (u.DN / u.s), " ")


# --------------------------------------------------------------------------- #
# σ and masks
# --------------------------------------------------------------------------- #


def test_sigma_needs_map_or_opt_in(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        _load(tmp_path, sigma=None)
    noise = DetectorNoise(gain=4.0, read_noise=40.0, n_reads=16, sky=1000.0)
    with pytest.raises(ValueError, match="exactly one"):
        _load(tmp_path, detector_noise=noise, sigma=FitsImage(tmp_path / "x.fits"))


def test_detector_noise_sigma_hand_computed(tmp_path):
    data = np.full((NY, NX), 500.0)
    data[0, 0] = -2000.0  # below the sky: only read noise is left
    noise = DetectorNoise(gain=4.0, read_noise=40.0, n_reads=16, sky=1000.0)
    std = _load(tmp_path, data=data, detector_noise=noise).target_data.std[0]
    # (4 e/DN * 1500 DN + 5 coadds * 40^2 e^2 / 16 reads) e^2, back to DN
    np.testing.assert_allclose(std[1, 1], np.sqrt(6500.0) / 4 * _per_dn(), rtol=1e-12)
    np.testing.assert_allclose(std[0, 0], np.sqrt(500.0) / 4 * _per_dn(), rtol=1e-12)


def test_bad_nonfinite_and_nonpositive_sigma_are_masked(tmp_path):
    data = np.full((NY, NX), 100.0)
    data[1, 1] = np.nan
    sigma = np.full((NY, NX), 4.0)
    sigma[2, 2] = 0.0
    bad = np.zeros((NY, NX))
    bad[3, 4] = 1
    exposure = _load(
        tmp_path,
        data=data,
        sigma=_write(tmp_path / "sig.fits", sigma),
        bad_pixels=_write(tmp_path / "bad.fits", bad),
    )
    expected = np.ones((NY, NX), bool)
    expected[1, 1] = expected[2, 2] = expected[3, 4] = False
    np.testing.assert_array_equal(exposure.target_data.mask[0], expected)
    assert exposure.target_data.nan_mask[0].sum() == NY * NX - 1
    assert exposure.diagnostics["masked_pixels"] == 3


def test_extension_selectors(tmp_path):
    primary = fits.PrimaryHDU(header=fits.Header(HEADER))
    sci = fits.ImageHDU(np.full((NY, NX), 100.0), name="SCI")
    err = fits.ImageHDU(np.full((NY, NX), 4.0), name="ERR")
    fits.HDUList([primary, sci, err]).writeto(tmp_path / "mef.fits")
    exposure = load_nirc2_exposure(
        FitsImage(tmp_path / "mef.fits", "SCI"),
        sigma=FitsImage(tmp_path / "mef.fits", 2),
        calibration=CAL,
        throughput=BAND,
        psf=DELTA,
        pre_2015=_solution(),
    )
    np.testing.assert_allclose(exposure.target_data.data[0], 100 * _per_dn())
    np.testing.assert_allclose(exposure.target_data.std[0], 4 * _per_dn())
    assert exposure.diagnostics["science"].endswith("mef.fits[SCI]")


# --------------------------------------------------------------------------- #
# Distortion and position angle
# --------------------------------------------------------------------------- #


def _offsets_mas(centers):
    """East and North offsets of the centres from CENTER, mas."""
    east, north = CENTER.spherical_offsets_to(centers)
    return east.to_value(u.mas), north.to_value(u.mas)


@pytest.mark.parametrize("subsample", [1, 2])
def test_distortion_moves_subpixel_centres(tmp_path, subsample):
    yy, xx = np.mgrid[0:NY, 0:NX].astype(float)

    def dx(x, y):
        return 0.4 + 0.03 * x - 0.02 * y

    def dy(x, y):
        return -0.7 + 0.01 * x + 0.05 * y

    exposure = _load(
        tmp_path,
        pre_2015=_solution(dx(xx, yy), dy(xx, yy)),
        subsample=subsample,
        position_angle=0 * u.deg,
    )
    east, north = _offsets_mas(exposure.target_data.subsample_centers[0])

    # raw position of sub-pixel [r, c]; keep those inside the map's pixel
    # centres, where bilinear interpolation of the linear maps is exact
    r, c = np.mgrid[0 : NY * subsample, 0 : NX * subsample]
    x, y = (c + 0.5) / subsample - 0.5, (r + 0.5) / subsample - 0.5
    inside = (x >= 0) & (x <= NX - 1) & (y >= 0) & (y <= NY - 1)
    # PA 0: +x is West, +y is North; raw reference pixel (7, 5) sits at CENTER
    expected_east = -SCALE_MAS * (x + dx(x, y) - 7 - dx(7, 5))
    expected_north = SCALE_MAS * (y + dy(x, y) - 5 - dy(7, 5))
    np.testing.assert_allclose(east[inside], expected_east[inside], atol=1e-6)
    np.testing.assert_allclose(north[inside], expected_north[inside], atol=1e-6)
    # without the maps the centres would sit at the raw positions
    assert np.abs(east[inside] + SCALE_MAS * (x[inside] - 7)).max() > 0.1 * SCALE_MAS


def test_position_angle_offset_applied(tmp_path):
    centers = _load(tmp_path).target_data.subsample_centers[0]
    # ROTPOSN - INSTANGL - beta = 30.7 - 0.7 - 0.25; 1e-4 deg allows for the
    # convergence of meridians over the 0.1 arcsec baseline
    up = centers[0, 7].position_angle(centers[NY - 1, 7]).to_value(u.deg)
    right = centers[5, 0].position_angle(centers[5, NX - 1]).to_value(u.deg)
    assert up == pytest.approx(29.75, abs=1e-4)
    assert right == pytest.approx(29.75 - 90 + 360, abs=1e-4)  # East to the left


def test_position_angle_override_and_rotator_mode(tmp_path):
    exposure = _load(
        tmp_path,
        header=HEADER | {"ROTMODE": "vertical angle"},
        position_angle=10 * u.deg,
    )
    centers = exposure.target_data.subsample_centers[0]
    up = centers[0, 7].position_angle(centers[NY - 1, 7]).to_value(u.deg)
    assert up == pytest.approx(10.0, abs=1e-4)
    with pytest.raises(ValueError, match="ROTMODE"):
        _load(tmp_path, header=HEADER | {"ROTMODE": "vertical angle"})


def test_date_selects_solution_and_override_wins(tmp_path):
    pre, post = _solution(name="pre"), _solution(name="post")
    assert select_distortion("2010-06-01", pre, post) is pre
    assert select_distortion("2015-04-13", pre, post) is pre
    assert select_distortion("2015-04-14T05:00:00", pre, post) is post
    with pytest.raises(ValueError, match="post_2015"):
        select_distortion("2019-01-01", pre, None)

    late = HEADER | {"DATE-OBS": "2019-01-01"}
    exposure = _load(tmp_path, header=late, pre_2015=pre, post_2015=post)
    assert exposure.diagnostics["distortion"] == "post, chosen by DATE-OBS 2019-01-01"
    override = _solution(name="mine")
    exposure = _load(tmp_path, header=late, distortion=override)
    assert exposure.diagnostics["distortion"] == "mine, chosen by override"


def test_published_solutions_read_maps(tmp_path):
    x = _write(tmp_path / "x.fits", np.ones((NY, NX))).path
    y = _write(tmp_path / "y.fits", np.zeros((NY, NX))).path
    pre, post = yelda2010(x, y), service2016(x, y)
    assert u.isclose(pre.pixel_scale, 9.952 * u.mas)
    assert u.isclose(pre.angle_offset, 0.252 * u.deg)
    assert u.isclose(post.pixel_scale, 9.971 * u.mas)
    assert u.isclose(post.angle_offset, 0.262 * u.deg)
    np.testing.assert_array_equal(pre.dx, 1.0)


def test_subarray_maps_raise(tmp_path):
    with pytest.raises(ValueError, match="distortion maps"):
        _load(tmp_path, pre_2015=_solution(shape=(NY + 1, NX)))


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #


def test_products_feed_the_jwst_target_response(tmp_path):
    exposure = _load(tmp_path, subsample=2)
    target = exposure.target_data
    assert target.data.shape == target.std.shape == target.mask.shape == (1, NY, NX)
    assert target.subsample_centers[0].shape == (2 * NY, 2 * NX)
    lam = np.sqrt((2.4**2 - 2.0**2) / 2 / np.log(1.2))  # tophat pivot, um
    assert u.isclose(exposure.meta.color, lam * u.um)
    assert exposure.throughput is BAND

    grid = WcsAstropy(CENTER, shape=60, fov=0.6 * u.arcsec)
    sky_domain = {"k": jft.ShapeWithDtype((60, 60))}
    response = build_jwst_response(
        sky_in_subsampled_data=build_sky_to_subsampled_data(
            sky_domain, target.subsample_centers, grid, LinearConfig(), None
        ),
        data_meta=exposure.meta,
        data_subsample=target.subsample,
        sky_meta=SkyMetaInformation(
            grid_extension=None, dvol=grid.dvol, unit=u.MJy / u.sr
        ),
        psf=target.psf,
        zero_flux_model=None,
        data_mask=target.mask,
    )
    # a flat 1 MJy/sr sky predicts 1 MJy/sr in every data pixel
    np.testing.assert_allclose(response({"k": jnp.ones((60, 60))}), 1.0, rtol=1e-6)
