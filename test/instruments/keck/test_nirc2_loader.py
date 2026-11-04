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
from jubik.instruments.jwst.likelihood.likelihood import (
    ShotNoiseGaussianLikelihoodBuilder,
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
GRID = WcsAstropy(CENTER, shape=60, fov=0.6 * u.arcsec)  # covers the data


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
    kwargs.setdefault("sky_grid", GRID)
    kwargs.setdefault("margin", 2)
    return load_nirc2_exposure(
        science, calibration=CAL, throughput=BAND, psf=DELTA, **kwargs
    )


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


def _response(exposure):
    """The existing JWST response (unit conversion, integration) of a sky
    ``{"k": MJy/sr on GRID}``."""
    target = exposure.target_data
    return build_jwst_response(
        sky_in_subsampled_data=build_sky_to_subsampled_data(
            {"k": jft.ShapeWithDtype(GRID.shape_yx)},
            target.subsample_centers,
            GRID,
            LinearConfig(),
            None,
        ),
        data_meta=exposure.meta,
        data_subsample=target.subsample,
        sky_meta=SkyMetaInformation(
            grid_extension=None, dvol=GRID.dvol, unit=u.MJy / u.sr
        ),
        psf=target.psf,
        zero_flux_model=None,
        data_mask=target.mask,
    )


def _predict(exposure):
    """The response for a flat 1 MJy/sr sky."""
    return np.asarray(_response(exposure)({"k": jnp.ones(GRID.shape_yx)}))


@pytest.mark.parametrize(
    "overrides, seconds",
    [
        ({}, 2.0 * 5),  # DN with the header's ITIME and COADDS
        ({"itime": 4.0, "coadds": 1}, 4.0),
        ({"bunit": "DN/s"}, 1.0),
    ],
)
def test_flat_sky_predicts_native_units(tmp_path, overrides, seconds):
    exposure = _load(tmp_path, subsample=2, **overrides)
    np.testing.assert_array_equal(exposure.target_data.data[0], 100.0)
    np.testing.assert_array_equal(exposure.target_data.std[0], 4.0)
    # 1 MJy/sr on a (10 mas)^2 pixel is 1e6 * PIXEL_SR Jy; 1e-6 Jy gives 1 DN/s
    expected = 1e6 * PIXEL_SR / 1e-6 * seconds
    np.testing.assert_allclose(_predict(exposure), expected, rtol=1e-6)


def test_meta_and_header_overrides(tmp_path):
    header = {k: v for k, v in HEADER.items() if k != "BUNIT"}
    exposure = _load(tmp_path, header=header, bunit="DN", itime=4.0, coadds=1)
    meta = exposure.meta
    assert meta.unit.physical_type == "spectral flux density"
    assert u.isclose(1 * meta.unit, 1e-6 / 4 * u.Jy)
    assert u.isclose(meta.pixel_scale, SCALE_MAS * u.mas)
    assert u.isclose(meta.dvol, (SCALE_MAS * u.mas) ** 2)
    assert u.isclose(exposure.diagnostics["jy_per_unit"], 1e-6 / 4 * u.Jy)


def test_missing_keyword_or_wrong_unit_raises(tmp_path):
    with pytest.raises(KeyError, match="ITIME"):
        _load(tmp_path, header={k: v for k, v in HEADER.items() if k != "ITIME"})
    with pytest.raises(ValueError, match="COADDS"):
        _load(tmp_path, coadds=2.7)
    with pytest.raises(KeyError, match="DATE-OBS.*distortion="):
        _load(tmp_path, header={k: v for k, v in HEADER.items() if k != "DATE-OBS"})
    with pytest.raises(ValueError, match="unit"):
        _load(tmp_path, bunit="MJy/sr")


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


@pytest.mark.parametrize("bunit, dn", [("DN", 1.0), ("DN/s", 2.0 * 5)])
def test_detector_noise_sigma_hand_computed(tmp_path, bunit, dn):
    data = np.full((NY, NX), 500.0) / dn  # 500 DN summed over the coadds
    data[0, 0] = -2000.0 / dn  # below the sky: only read noise is left
    noise = DetectorNoise(gain=4.0, read_noise=40.0, n_reads=16, sky=1000.0)
    std = _load(tmp_path, data=data, detector_noise=noise, bunit=bunit)
    std = std.target_data.std[0] * dn
    # (4 e/DN * 1500 DN + 5 coadds * 40^2 e^2 / 16 reads) e^2, back to DN
    np.testing.assert_allclose(std[1, 1], np.sqrt(6500.0) / 4, rtol=1e-12)
    np.testing.assert_allclose(std[0, 0], np.sqrt(500.0) / 4, rtol=1e-12)


def test_detector_noise_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        DetectorNoise(gain=np.nan, read_noise=40.0, n_reads=16, sky=0.0)
    noise = DetectorNoise(gain=4.0, read_noise=40.0, n_reads=16, sky=np.zeros(NX))
    with pytest.raises(ValueError, match="sky"):
        _load(tmp_path, detector_noise=noise)


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


def test_cut_to_the_sky_grid_and_pixels_outside_it_masked(tmp_path):
    small = WcsAstropy(CENTER, shape=5, fov=50 * u.mas)  # +-25 mas
    exposure = _load(tmp_path, sky_grid=small, position_angle=0 * u.deg)
    # the grid spans raw x = 4.5..9.5, y = 2.5..7.5; plus the margin of 2
    r0, r1, c0, c1 = exposure.diagnostics["bounds"]
    assert r0 <= 1 and r1 >= 10 and c0 <= 3 and c1 >= 12
    assert c1 - c0 < NX
    target = exposure.target_data
    assert target.data.shape[1:] == target.subsample_centers[0].shape
    assert target.data.shape[1:] == (r1 - r0, c1 - c0)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    # PA 0 without distortion: pixel offsets from CENTER are 10 mas steps
    east, north = -SCALE_MAS * (xx - 7), SCALE_MAS * (yy - 5)
    near = (np.abs(east) <= 10) & (np.abs(north) <= 10)
    far = (np.abs(east) >= 40) | (np.abs(north) >= 40)
    mask = exposure.target_data.mask[0]
    assert mask[near].all() and not mask[far].any()
    assert exposure.diagnostics["masked_pixels"] >= far.sum()


def test_every_pixel_masked_raises(tmp_path):
    with pytest.raises(ValueError, match="every pixel"):
        _load(tmp_path, bad_pixels=_write(tmp_path / "bad.fits", np.ones((NY, NX))))


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
        sky_grid=GRID,
        margin=2,
        pre_2015=_solution(),
    )
    np.testing.assert_array_equal(exposure.target_data.data[0], 100.0)
    np.testing.assert_array_equal(exposure.target_data.std[0], 4.0)
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


def test_header_pointing_frame(tmp_path):
    fk5 = HEADER | {"RADESYS": "FK5"}
    centers = _load(tmp_path, header=fk5, position_angle=0 * u.deg)
    centers = centers.target_data.subsample_centers[0]
    # raw reference pixel (7, 5) sits at CRVAL read in FK5 J2000
    reference = SkyCoord(150 * u.deg, 30 * u.deg, frame="fk5").icrs
    assert centers[5, 7].separation(reference) < 1e-6 * u.mas


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
    with pytest.raises(KeyError, match="ROTMODE"):
        _load(tmp_path, header={k: v for k, v in HEADER.items() if k != "ROTMODE"})


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


def test_products_shapes_and_band(tmp_path):
    exposure = _load(tmp_path, subsample=2.0)
    target = exposure.target_data
    assert target.data.shape == target.std.shape == target.mask.shape == (1, NY, NX)
    assert target.psf.shape == (1, *DELTA.shape)
    assert target.subsample == 2
    assert target.subsample_centers[0].shape == (2 * NY, 2 * NX)
    lam = np.sqrt((2.4**2 - 2.0**2) / 2 / np.log(1.2))  # tophat pivot, um
    assert u.isclose(exposure.meta.color, lam * u.um)
    assert exposure.throughput is BAND


# --------------------------------------------------------------------------- #
# Shot noise
# --------------------------------------------------------------------------- #

GAIN, READ_NOISE, N_READS, COADDS = 4.0, 40.0, 16, HEADER["COADDS"]


def _mcds(rng, draws, electrons_per_coadd):
    """DN summed over the coadds of MCDS reads: N_READS reads at each end of
    every coadd, each with READ_NOISE / sqrt(2) (READ_NOISE is per CDS)."""
    electrons = np.asarray(electrons_per_coadd, float)
    shape = (draws, *electrons.shape, COADDS)

    def group():
        return rng.normal(0, READ_NOISE / np.sqrt(2), (*shape, N_READS)).mean(-1)

    end = rng.poisson(electrons[..., None], shape) + group()
    return (end - group()).sum(-1) / GAIN


@pytest.mark.parametrize("bunit, dn", [("DN", 1.0), ("DN/s", 2.0 * 5)])
def test_detector_noise_matches_monte_carlo_mcds(tmp_path, bunit, dn):
    flat = np.ones((NY, NX))
    flat[1, 2], flat[3, 4] = 0.7, 1.4
    sky = 800.0  # DN summed over coadds, in the flat-fielded image
    detector = DetectorNoise(GAIN, READ_NOISE, N_READS, sky, flat)
    noise = _load(tmp_path, bunit=bunit, detector_noise=detector).noise
    rng = np.random.default_rng(172)
    # science DN summed over coadds: faint below the sky, bright above it
    for (i, j), signal in (((1, 2), 30.0), ((3, 4), 3e4), ((0, 0), -200.0)):
        f = flat[i, j]
        raw = _mcds(rng, 40_000, GAIN * f * (signal + sky) / COADDS)
        science = (raw / f - sky) / dn  # flat-fielded, sky subtracted, native
        predicted = noise.a[i, j] + noise.b[i, j] * (signal + sky) / dn
        assert noise.sky * dn == pytest.approx(sky)
        # the variance of 40000 draws is good to 0.7 %
        assert science.var() == pytest.approx(predicted, rel=0.03), (i, j)
        assert science.mean() == pytest.approx(signal / dn, rel=0.02, abs=0.5 / dn)


def test_flat_enters_the_plain_sigma_and_is_checked(tmp_path):
    data = np.full((NY, NX), 500.0)
    flat = np.full((NY, NX), 2.0)
    detector = DetectorNoise(GAIN, READ_NOISE, N_READS, 1000.0, flat)
    std = _load(tmp_path, data=data, detector_noise=detector).target_data.std[0]
    # (1500 DN / (4 e/DN 2) + 5 coadds 40^2 e^2 / (16 reads 4^2 2^2)) DN^2
    np.testing.assert_allclose(std, np.sqrt(1500 / 8 + 500 / 64), rtol=1e-12)
    assert _load(tmp_path).noise is None  # a σ map has no model
    with pytest.raises(ValueError, match="flat"):
        DetectorNoise(GAIN, READ_NOISE, N_READS, 0.0, -flat)
    with pytest.raises(ValueError, match="read_noise > 0"):
        DetectorNoise(GAIN, 0.0, N_READS, 0.0)
    flat[1, 1], flat[2, 2] = 0.0, np.nan  # dead pixels of a real flat
    detector = DetectorNoise(GAIN, READ_NOISE, N_READS, 1000.0, flat)
    mask = _load(tmp_path, data=data, detector_noise=detector).target_data.mask[0]
    assert not mask[1, 1] and not mask[2, 2] and mask.sum() == mask.size - 2
    with pytest.raises(ValueError, match="flat"):
        _load(
            tmp_path,
            detector_noise=DetectorNoise(GAIN, READ_NOISE, N_READS, 0.0, flat[:2]),
        )


def test_normalized_residuals_through_the_response(tmp_path):
    sky = 1000.0
    detector = DetectorNoise(GAIN, READ_NOISE, N_READS, sky)
    exposure = _load(tmp_path, detector_noise=detector)
    response = _response(exposure)
    per_unit_sky = _predict(exposure).mean()  # DN of 1 MJy/sr
    field = np.full(GRID.shape_yx, 20.0 / per_unit_sky)  # 20 DN, below the sky
    field[:, 30:] = 3e4 / per_unit_sky  # 30000 DN, shot-noise dominated
    builder = ShotNoiseGaussianLikelihoodBuilder(
        response, exposure.target_data.data, exposure.noise, exposure.target_data.mask
    )
    model, inverse_std = (np.asarray(v) for v in builder.model()({"k": field}))
    rng = np.random.default_rng(172)
    data = _mcds(rng, 300, GAIN * (model + sky) / COADDS) - sky
    no_source_shot = 1 / np.sqrt(exposure.noise.a + exposure.noise.b * sky)
    for pixels in (model < 100, model > 1e4):
        assert pixels.sum() > 30
        normalized = (data[:, pixels] - model[pixels]) * inverse_std[pixels]
        assert normalized.mean() == pytest.approx(0, abs=0.03)
        assert normalized.var() == pytest.approx(1, abs=0.04)
    bright = (data[:, model > 1e4] - model[model > 1e4]) * no_source_shot
    assert bright.var() > 10
