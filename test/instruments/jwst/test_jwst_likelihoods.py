"""Smoke test for `build_jwst_likelihoods`.

Runs the full chain, config parsing, preload, load, response, likelihood, on a
fake observation. Two production seams are patched:

- The `JwstData` name in the preloader and the three loaders, rebound to
  `FakeJwstData`, which is a `JwstData` with an astropy WCS and in-memory arrays
  instead of a `jwst` datamodel.
- `build_webb_psf` in `psf/jwst_kernel.py`, replaced by a delta kernel so that
  stpsf is never imported. `load_psf_kernel` still writes and reads the `.npy`.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jubik.color import Color
from jubik.grid import Grid
from jubik.instruments.jwst.alignment import star_model
from jubik.instruments.jwst.data.jwst_data import DataMetaInformation, JwstData
from jubik.instruments.jwst.data.jwst_information import (
    get_dvol,
    get_pixel_distance,
)
from jubik.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik.instruments.jwst.parse.alignment.star_alignment import (
    StarAlignmentConfig,
)
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.wcs import WcsAstropy

FILTER = "F444W"
CENTER = SkyCoord(ra=10.05, dec=-4.95, unit="deg")
DATA_SHAPE = (64, 64)
NAN_PIXEL = (32, 30)
GRID_SHAPE = (16, 16)
GRID_FOV = [1.6, 1.6] * u.arcsec
SKY_DOMAIN = {"sky": jft.ShapeWithDtype((1,) + GRID_SHAPE, float)}

JWST_DATA_SITES = (
    "jubik.instruments.jwst.data.preloader.preloader.JwstData",
    "jubik.instruments.jwst.data.loader.data_loader.JwstData",
    "jubik.instruments.jwst.data.loader.target_loader.JwstData",
    "jubik.instruments.jwst.data.loader.stars_loader.JwstData",
)


class FakeJwstData(JwstData):
    """`JwstData` backed by arrays and an astropy WCS instead of a datamodel."""

    def __init__(self, filepath: str):
        pixel_scale = get_pixel_distance(FILTER)
        rng = np.random.default_rng(abs(hash(str(filepath))) % 2**32)
        data = rng.normal(size=DATA_SHAPE)
        data[NAN_PIXEL] = np.nan
        self.dm = SimpleNamespace(
            data=data,
            err=np.ones(DATA_SHAPE),
            meta=SimpleNamespace(
                date="2023-01-01T00:00:00.000",
                observation=SimpleNamespace(date_beg="2023-01-01T00:00:00.000"),
                pointing=SimpleNamespace(ra_v1=CENTER.ra.deg, dec_v1=CENTER.dec.deg),
                wcsinfo=SimpleNamespace(ra_ref=CENTER.ra.deg, dec_ref=CENTER.dec.deg),
                instrument=SimpleNamespace(detector="NRCALONG"),
                subarray=SimpleNamespace(xstart=1, ystart=1),
            ),
        )
        self.wcs = WcsAstropy(
            center=CENTER,
            shape=DATA_SHAPE,
            fov=u.Quantity([pixel_scale * s for s in DATA_SHAPE]),
        )
        self.shape = DATA_SHAPE
        self.filter = FILTER
        self.camera = "NIRCAM"
        self.meta = DataMetaInformation(
            unit=u.Unit("MJy/sr"),
            dvol=get_dvol(FILTER),
            pixel_scale=pixel_scale,
            color=self.pivot_wavelength,
        )


class MissingExposureTimeJwstData(FakeJwstData):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.dm.meta.observation.date_beg = None
        self.dm.meta.exposure = SimpleNamespace(start_time=None)


class NanStarsJwstData(FakeJwstData):
    """`FakeJwstData` with NaN pixels under every `fake_gaia` star cutout.

    Both stars land inside rows and columns 8 to 40 with a 9 pixel window, so
    `load_one_stars_bundle` rejects each of them as an all-NaN cutout while the
    target still has unmasked pixels.
    """

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.dm.data[8:40, 8:40] = np.nan


def fake_build_webb_psf(camera, filter, center_pixel, webbpsf_path, subsample, *a, **k):
    size = 5 * subsample
    psf = np.zeros((size, size))
    psf[size // 2, size // 2] = 1.0
    return psf


@pytest.fixture
def patched_seams(monkeypatch):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, FakeJwstData)
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", fake_build_webb_psf)


def make_grid():
    spatial = WcsAstropy(center=CENTER, shape=GRID_SHAPE, fov=GRID_FOV)
    # one channel covering the whole fake F444W curve
    return Grid(spatial=spatial, spectral=Color([3.6, 5.2] * u.um))


def write_fake_throughput(tmp_path):
    """A smooth F444W-like curve, 3.7 to 5.1 um, in the STScI file format."""
    throughput_dir = tmp_path / "throughputs"
    throughput_dir.mkdir(exist_ok=True)
    wavelength_um = np.linspace(3.7, 5.1, 141)
    throughput = 0.5 * np.sin(np.pi * (wavelength_um - 3.7) / 1.4) ** 2
    np.savetxt(
        throughput_dir / f"{FILTER}_mean_system_throughput.txt",
        np.c_[wavelength_um, throughput],
        header="Microns Throughput",
        comments="",
    )
    return throughput_dir


def make_config(
    tmp_path,
    n_files=1,
    gaia=False,
    variable_covariance=False,
    zero_flux=False,
    gaia_subsample=2,
    fov="0.5arcsec",
    calibration=None,
):
    files = [str(tmp_path / f"fake_{ii}.fits") for ii in range(n_files)]
    telescope = {
        "psf": {
            "webbpsf_path": "",
            "psf_library": str(tmp_path / "psf_library"),
            "psf_arcsec_extension": 0.4,
        },
        "target": {"subsample": 2},
        "rotation_and_shift": {
            "linear": {"order": 1, "mode": "constant"},
            "correction_priors": {
                "model": "shift",
                "shift_unit": "arcsec",
                "rotation_unit": "deg",
                "default": {
                    "rotation": ["delta", 0.0, 0.1],
                    "shift": ["normal", 0.0, 0.1],
                },
            },
        },
    }
    if gaia:
        (tmp_path / "gaia").mkdir(exist_ok=True)
        telescope["gaia_alignment"] = {
            "fov": fov,
            "subsample": gaia_subsample,
            "star_light": ["lognormal", 1.0, 1.0],
            "library_path": str(tmp_path / "gaia"),
        }
    if zero_flux:
        telescope["zero_flux"] = {"default": ["lognormal", 0.9, 4]}
    if calibration is not None:
        telescope["calibration"] = calibration
    if variable_covariance:
        telescope["variable_covariance"] = {
            "additive_std_value": {
                "shape_type": "pixel",
                "distribution": ["invgamma", 4, 0.01, 0.0],
            }
        }
    files_block = {
        "filter": {FILTER.lower(): files},
        "throughputs": str(write_fake_throughput(tmp_path)),
    }
    return {"files": files_block, "telescope": telescope}


def evaluate(likelihood):
    x = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
    return float(likelihood(x))


def test_minimal_config_builds_and_evaluates(patched_seams, tmp_path):
    cfg = make_config(tmp_path)
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)

    assert products.alignment is None
    assert len(products.target.likelihoods) == 1
    target = products.target.likelihoods[0]
    assert target.filter == FILTER.lower()

    builder = target.builder
    assert builder.data.shape[0] == 1
    assert builder.data.shape == builder.mask.shape == builder.std.shape
    assert builder.mask.sum() > 0
    assert not np.isnan(builder.data[builder.mask]).any()

    sky_filter = products.target.sky_filter
    np.testing.assert_allclose(sky_filter.bands[FILTER.lower()].weights, [1.0])
    assert sky_filter.target[FILTER.lower()].shape == GRID_SHAPE

    likelihood = products.target.likelihood
    assert "sky" in likelihood.domain
    assert np.isfinite(evaluate(likelihood))

    saved = list((tmp_path / "psf_library").glob("*.npy"))
    assert len(saved) == 1
    assert saved[0].name.startswith("nircam_f444w_nrcalong_")


def test_missing_exposure_time_without_gaia_alignment(monkeypatch, patched_seams, tmp_path):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, MissingExposureTimeJwstData)

    products = build_jwst_likelihoods(make_config(tmp_path), make_grid(), SKY_DOMAIN)

    assert products.alignment is None


def test_missing_exposure_time_with_opd_by_date_raises(
    monkeypatch, patched_seams, tmp_path
):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, MissingExposureTimeJwstData)
    cfg = make_config(tmp_path)
    cfg["telescope"]["psf"]["opd_by_date"] = True

    with pytest.raises(ValueError, match="`opd_by_date` needs the exposure start"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def test_two_files_share_bounds(patched_seams, tmp_path):
    cfg = make_config(tmp_path, n_files=2)
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)

    builder = products.target.likelihoods[0].builder
    assert builder.data.shape[0] == 2
    assert not np.allclose(builder.data[0], builder.data[1], equal_nan=True)
    assert np.isfinite(evaluate(products.target.likelihood))


@pytest.mark.parametrize("zero_flux", [True, False])
def test_gaia_and_variable_covariance(patched_seams, fake_gaia, tmp_path, zero_flux):
    calls, table = fake_gaia
    cfg = make_config(
        tmp_path, gaia=True, variable_covariance=True, zero_flux=zero_flux
    )
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)

    assert len(calls) == 1
    assert products.alignment is not None
    assert len(products.alignment.likelihood.likelihoods) == 1

    target = products.target.likelihood
    minimal = build_jwst_likelihoods(make_config(tmp_path), make_grid(), SKY_DOMAIN)
    extra_keys = set(target.domain.tree) - set(minimal.target.likelihood.domain.tree)
    assert extra_keys, "variable covariance should add parameters"
    assert any("zero_flux" in k for k in extra_keys) == zero_flux
    assert np.isfinite(evaluate(target))

    alignment = products.alignment.likelihood.likelihood
    keys = set(alignment.domain.tree)
    for star_id in table["source_id"]:
        assert f"{FILTER.lower()}_{star_id}_brightness" in keys
    assert any("zero_flux" in k for k in keys) == zero_flux
    assert np.isfinite(evaluate(alignment))


def test_fov_mapping_picks_the_filter_entry(patched_seams, fake_gaia, tmp_path):
    # The filter's own entry wins over `default`, so the star cutouts come out
    # as with the plain "0.5arcsec".
    fov = {"default": "9arcsec", FILTER.lower(): "0.5arcsec"}
    mapped = build_jwst_likelihoods(make_config(tmp_path, gaia=True, fov=fov), make_grid(), SKY_DOMAIN)
    plain = build_jwst_likelihoods(make_config(tmp_path, gaia=True), make_grid(), SKY_DOMAIN)
    cutout = mapped.alignment.plotting.psf[0].data[0]
    assert cutout.shape == plain.alignment.plotting.psf[0].data[0].shape
    assert np.isfinite(evaluate(mapped.alignment.likelihood.likelihood))


def test_fov_mapping_needs_a_default():
    from jubik.instruments.jwst.parse.alignment.star_alignment import (
        StarAlignmentConfig,
    )

    raw = {"fov": {"F444W": "0.5arcsec"}, "subsample": 2, "star_light": ["lognormal", 1.0, 1.0]}
    with pytest.raises(ValueError, match="needs a `default` entry"):
        StarAlignmentConfig.from_yaml_dict(raw)
    raw["fov"]["default"] = "1arcsec"
    config = StarAlignmentConfig.from_yaml_dict(raw)
    assert config.fov_of("F444W") == 0.5 * u.arcsec
    assert config.fov_of("f356w") == 1 * u.arcsec


@pytest.mark.parametrize(
    "fov, match",
    [
        ({"default": "1arcsec", "F44W": "0.5arcsec"}, "unknown filter names"),
        ({"default": "1arcsec", "F444W": "1m"}, r"fov\.F444W: needs an angle"),
        ({"default": "1arcsec", "F444W": "wide"}, r"fov\.F444W: needs an angle"),
        ("0.5", "fov: needs an angle"),
    ],
)
def test_fov_rejects_typos_and_non_angles(fov, match):
    from jubik.instruments.jwst.parse.alignment.star_alignment import (
        StarAlignmentConfig,
    )

    raw = {"fov": fov, "subsample": 2, "star_light": ["lognormal", 1.0, 1.0]}
    with pytest.raises(ValueError, match=match):
        StarAlignmentConfig.from_yaml_dict(raw)


def test_fit_star_positions_adds_one_shared_latent_per_star(
    patched_seams, fake_gaia, tmp_path
):
    _, table = fake_gaia
    cfg = make_config(tmp_path, gaia=True)
    gaia = cfg["telescope"]["gaia_alignment"]
    assert StarAlignmentConfig.from_yaml_dict(gaia).fit_star_positions is True
    gaia["fit_star_positions"] = False
    plain = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)
    plain_keys = plain.alignment.likelihood.likelihood.domain.tree
    assert not any("astrometry" in k for k in plain_keys)

    gaia["fit_star_positions"] = True
    alignment = build_jwst_likelihoods(
        cfg, make_grid(), SKY_DOMAIN
    ).alignment.likelihood.likelihood
    keys = alignment.domain.tree
    # No filter prefix, so every filter shares the star's latent.
    assert {k for k in keys if "astrometry" in k} == {
        f"star_{star_id}_astrometry" for star_id in table["source_id"]
    }
    for star_id in table["source_id"]:
        assert keys[f"star_{star_id}_astrometry"].shape == (4,)
    assert np.isfinite(evaluate(alignment))


def test_fit_star_positions_moves_the_star_in_the_likelihood(
    patched_seams, fake_gaia, tmp_path
):
    _, table = fake_gaia
    # 0.1 arcsec errors, so a unit latent moves the star by about a pixel.
    for name in ("ra_error", "dec_error"):
        table[name] = [100.0, 100.0] * u.mas
    cfg = make_config(tmp_path, gaia=True)
    cfg["telescope"]["gaia_alignment"]["fit_star_positions"] = True
    alignment = build_jwst_likelihoods(
        cfg, make_grid(), SKY_DOMAIN
    ).alignment.likelihood.likelihood
    x = jft.random_like(jax.random.PRNGKey(0), alignment.domain)
    moved = x.tree | {"star_1_astrometry": x.tree["star_1_astrometry"] + 1.0}
    assert not np.isclose(float(alignment(x)), float(alignment(jft.Vector(moved))))


def test_fit_star_positions_needs_the_gaia_covariance(
    patched_seams, fake_gaia, tmp_path
):
    _, table = fake_gaia
    table["pmdec_error"][0] = np.nan
    cfg = make_config(tmp_path, gaia=True)
    cfg["telescope"]["gaia_alignment"]["fit_star_positions"] = True
    with pytest.raises(ValueError, match="Gaia star 1 has no astrometric covariance"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


class LaterSecondExposureJwstData(FakeJwstData):
    """`fake_1` five months after `fake_0`; `fake_0` has NaN pixels under the
    second `fake_gaia` star, so that star is only in `fake_1`."""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        if str(filepath).endswith("fake_1.fits"):
            self.dm.meta.observation.date_beg = "2023-06-01T00:00:00.000"
        else:
            star_2 = SkyCoord(10.0503 * u.deg, -4.9503 * u.deg)
            x, y = np.round(self.wcs.world_to_pixel(star_2)).astype(int)
            self.dm.data[y - 6 : y + 7, x - 6 : x + 7] = np.nan


def test_fit_star_positions_uses_each_exposure_epoch(
    monkeypatch, patched_seams, fake_gaia, tmp_path
):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, LaterSecondExposureJwstData)
    epochs = {}

    def spy(star, times):
        epochs.setdefault(int(star.id), []).append([Time(t).isot for t in times])
        return original(star, times)

    original = star_model._star_position_offset
    monkeypatch.setattr(star_model, "_star_position_offset", spy)
    cfg = make_config(tmp_path, n_files=2, gaia=True)
    cfg["telescope"]["gaia_alignment"]["fit_star_positions"] = True
    alignment = build_jwst_likelihoods(
        cfg, make_grid(), SKY_DOMAIN
    ).alignment.likelihood.likelihood
    # Once per star: the psf and smoothed-psf responses share the star model.
    assert epochs == {
        1: [["2023-01-01T00:00:00.000", "2023-06-01T00:00:00.000"]],
        2: [["2023-06-01T00:00:00.000"]],
    }
    assert np.isfinite(evaluate(alignment))


def test_empty_gaia_catalog_raises(patched_seams, fake_gaia, tmp_path):
    _, table = fake_gaia
    table.remove_rows(slice(None))
    cfg = make_config(tmp_path, gaia=True)
    with pytest.raises(ValueError, match="no usable star.*0 catalog stars"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def test_all_gaia_stars_rejected_raises(
    monkeypatch, patched_seams, fake_gaia, tmp_path
):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, NanStarsJwstData)
    cfg = make_config(tmp_path, gaia=True)
    with pytest.raises(ValueError, match="2 catalog stars, 0 with a valid cutout"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


@pytest.mark.parametrize("subsample", [1, 2, 3])
def test_gaia_stars_on_edges_and_corners(
    monkeypatch, patched_seams, fake_gaia, tmp_path, subsample
):
    from jubik.instruments.jwst import jwst_likelihoods

    _, table = fake_gaia
    table.remove_rows(slice(None))
    # Pixel (x, y) positions on every edge, every corner and the interior.
    xy = [(x, y) for x in (0.2, 31.6, 63.3) for y in (-0.4, 30.7, 63.4)]
    # Off the detector by a few pixels (inside the Gaia circle): must be dropped.
    outside = [(66.0, 30.0), (30.0, -0.6), (-2.0, -2.0)]
    wcs = FakeJwstData("").wcs
    for ii, position in enumerate(wcs.pixel_to_world(*np.array(xy + outside).T)):
        table.add_row(
            (ii + 1, position.ra, position.dec, 0.0, 0.0) + (0.1,) * 4 + (0.0,) * 6
        )

    loaded, load_data = [], jwst_likelihoods.load_data
    monkeypatch.setattr(
        jwst_likelihoods,
        "load_data",
        lambda *a, **k: loaded.append(load_data(*a, **k)) or loaded[-1],
    )

    n_files = 2
    cfg = make_config(tmp_path, n_files=n_files, gaia=True, gaia_subsample=subsample)
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)

    stars_data = loaded[0].stars_data
    assert sorted(stars_data) == list(range(1, len(xy) + 1))
    for star in stars_data.values():
        n_obs, fov, fov_x = star.data.shape
        assert (n_obs, fov_x) == (n_files, fov) and fov % 2 == 1
        assert star.mask.dtype == bool and star.mask.any()
        assert np.isfinite(star.std[star.mask]).all()
        # Fine pixel k covers [k - 0.5, k + 0.5) of the subsampled cutout.
        assert (star.star_in_subsampled_pixels >= -0.5).all()
        assert (star.star_in_subsampled_pixels < fov * subsample - 0.5).all()
    assert np.isfinite(evaluate(products.alignment.likelihood.likelihood))


def test_sky_domain_must_match_grid(patched_seams, tmp_path):
    sky_domain = {"sky": jft.ShapeWithDtype((2,) + GRID_SHAPE, float)}
    with pytest.raises(ValueError, match="does not match the grid"):
        build_jwst_likelihoods(make_config(tmp_path), make_grid(), sky_domain)


def test_file_filter_must_match_config_key(patched_seams, tmp_path):
    cfg = make_config(tmp_path)
    throughput_dir = Path(cfg["files"]["throughputs"])
    shutil.copy(
        throughput_dir / f"{FILTER}_mean_system_throughput.txt",
        throughput_dir / "F277W_mean_system_throughput.txt",
    )
    cfg["files"]["filter"] = {"f277w": cfg["files"]["filter"].pop(FILTER.lower())}
    with pytest.raises(ValueError, match=f"observation is {FILTER}.*f277w"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def broad_build_webb_psf(
    camera, filter, center_pixel, webbpsf_path, subsample, *a, **k
):
    """A Gaussian core with power-law wings, 0.3" wide, so sky light near the sky
    edge reaches unmasked pixels through the PSF."""
    size = 9 * subsample + 1
    r = np.hypot(*np.mgrid[:size, :size] - (size - 1) / 2)
    psf = np.exp(-0.5 * (r / subsample) ** 2) + 0.01 / (1 + (r / subsample) ** 3)
    return psf / psf.sum()


def rotated_jwst_data(angle_deg):
    """`FakeJwstData` on a detector turned by `angle_deg` against the sky grid."""

    class RotatedJwstData(FakeJwstData):
        def __init__(self, filepath: str):
            super().__init__(filepath)
            self.wcs = WcsAstropy(
                center=CENTER,
                shape=DATA_SHAPE,
                fov=u.Quantity([get_pixel_distance(FILTER) * s for s in DATA_SHAPE]),
                position_angle=angle_deg * u.deg,
            )

    return RotatedJwstData


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize(
    "algorithm",
    [{"linear": {"order": 1, "mode": "constant"}}, {"nufft": {"mode": "constant"}}],
)
def test_data_extension_crops_the_cutout_without_changing_the_likelihood(
    patched_seams, monkeypatch, tmp_path, algorithm, rotated
):
    if "nufft" in algorithm:
        pytest.importorskip("jax_finufft")
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", broad_build_webb_psf)
    if rotated:
        for site in JWST_DATA_SITES:
            monkeypatch.setattr(site, rotated_jwst_data(30))

    def build(extension):
        cfg = make_config(tmp_path, zero_flux=True)
        cfg["telescope"]["psf"]["psf_arcsec_extension"] = 1.4
        # a 0.02" shift prior: the margin must hold the sky moved by 3 sigma
        priors = cfg["telescope"]["rotation_and_shift"]["correction_priors"]
        priors["default"]["shift"] = ["normal", 0.0, 0.02]
        cfg["telescope"]["rotation_and_shift"] |= algorithm
        cfg["telescope"]["rotation_and_shift"].pop(
            "nufft" if "linear" in algorithm else "linear", None
        )
        if extension is not None:
            cfg["telescope"]["target"]["data_arcsec_extension"] = extension
        return build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN).target

    # the PSF-sized cutout (the opt-out), an explicit margin, the automatic one
    wide, cropped, auto = build("off"), build(0.4), build(None)
    bw = wide.likelihoods[0].builder
    x = jft.random_like(jax.random.PRNGKey(1), wide.likelihood.domain).tree
    # a nonzero shift latent, and a bright sky edge row, so a crop that dropped
    # sky light would show
    assert np.abs(x[f"{FILTER.lower()}_shift"]).max() > 1
    x = x | {"sky": x["sky"].at[..., 0, :].add(5.0)}
    for small in (cropped, auto):
        bc = small.likelihoods[0].builder
        assert bc.data.shape[-1] < bw.data.shape[-1]
        np.testing.assert_array_equal(bc.data[bc.mask], bw.data[bw.mask])
        np.testing.assert_allclose(
            float(small.likelihood(x)), float(wide.likelihood(x)), rtol=1e-12
        )


# 0.2" holds the sky at zero shift, not when the 0.1" shift prior moves it by 3
# sigma; a negative margin is refused outright, however large
@pytest.mark.parametrize("extension", [0.0, 0.2, -0.1, -5.0])
def test_data_extension_too_small_raises(patched_seams, tmp_path, extension):
    cfg = make_config(tmp_path)
    cfg["telescope"]["target"]["data_arcsec_extension"] = extension
    with pytest.raises(ValueError, match="data_arcsec_extension"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


@pytest.mark.parametrize(
    "model, shift, rotation, raises",
    [
        ("shift", ["delta", 0.0, 0.0], ["delta", 0.0, 0.0], False),
        ("shift", ["uniform", -0.3, 0.3], ["delta", 0.0, 0.0], True),
        ("shift", ["uniform", -0.3, 0.0], ["delta", 0.0, 0.0], True),  # one-sided
        ("shift", ["delta", 0.3, 0.0], ["delta", 0.0, 0.0], True),  # a known shift
        # about the grid centre (the fake boresight) 3 x 5 deg moves the
        # corners, 1.1" out, by 0.3"
        ("rshift", ["delta", 0.0, 0.0], ["normal", 0.0, 5.0], True),
        ("rshift", ["delta", 0.0, 0.0], ["delta", 0.0, 0.0], False),
    ],
)
def test_data_extension_holds_the_prior_reach(
    patched_seams, tmp_path, model, shift, rotation, raises
):
    cfg = make_config(tmp_path, n_files=2)
    priors = cfg["telescope"]["rotation_and_shift"]["correction_priors"]
    priors["model"] = model
    priors["default"] = {"shift": shift, "rotation": rotation}
    cfg["telescope"]["target"]["data_arcsec_extension"] = 0.4
    if raises:
        with pytest.raises(ValueError, match="data_arcsec_extension"):
            build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)
    else:
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


@pytest.mark.parametrize(
    "algorithm",
    [{"linear": {"order": 1, "mode": "wrap"}}, {"nufft": {"mode": "wrap"}}],
)
def test_data_extension_needs_mode_constant(patched_seams, tmp_path, algorithm):
    # outside constant mode the sky reads nonzero beyond the grid, so no margin holds
    # it: an explicit and the automatic margin are refused, the opt-out builds
    cfg = make_config(tmp_path)
    cfg["telescope"]["rotation_and_shift"].pop("linear")
    cfg["telescope"]["rotation_and_shift"] |= algorithm
    for extension in (1.0, "absent"):
        target = cfg["telescope"]["target"]
        target.pop("data_arcsec_extension", None)
        if extension != "absent":
            target["data_arcsec_extension"] = extension
        with pytest.raises(ValueError, match="mode 'constant'"):
            build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)
    cfg["telescope"]["target"]["data_arcsec_extension"] = "off"
    build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def test_data_extension_off_is_the_psf_sized_cutout(patched_seams, tmp_path):
    # off, false and null all give the cutout of psf_arcsec_extension (0.4" here),
    # which an explicit 0.4 reproduces once a delta shift prior lets it pass the check
    def shape(value, shift=None):
        cfg = make_config(tmp_path)
        cfg["telescope"]["target"]["data_arcsec_extension"] = value
        if shift is not None:
            priors = cfg["telescope"]["rotation_and_shift"]["correction_priors"]
            priors["default"]["shift"] = shift
        target = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN).target
        return target.likelihoods[0].builder.data.shape

    psf = shape(0.4, shift=["delta", 0.0, 0.0])
    assert shape("off") == shape(False) == shape(None) == psf


def test_automatic_data_extension_follows_the_shift_prior(patched_seams, tmp_path):
    def width(sigma):
        cfg = make_config(tmp_path)
        priors = cfg["telescope"]["rotation_and_shift"]["correction_priors"]
        priors["default"]["shift"] = ["normal", 0.0, sigma]
        target = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN).target
        return target.likelihoods[0].builder.data.shape[-1]

    # 3 sigma more reach on each side, times sqrt(2) for any detector angle, in
    # 0.063" data pixels
    expected = 2 * 3 * 0.08 * np.sqrt(2) / 0.063
    assert width(0.1) - width(0.02) == pytest.approx(expected, abs=2)


@pytest.mark.parametrize("angle", [30, 60])
def test_automatic_data_extension_on_a_rotated_detector(
    patched_seams, monkeypatch, tmp_path, angle
):
    # the check widens the sky along its own axes, which on a turned detector reach
    # further than the same width along the detector's: the default must still build
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, rotated_jwst_data(angle))
    build_jwst_likelihoods(make_config(tmp_path), make_grid(), SKY_DOMAIN)


def test_automatic_data_extension_is_checked(patched_seams, tmp_path):
    # the automatic margin is checked too: a per-filter bounds adjustment that eats it
    # raises instead of dropping sky light
    cfg = make_config(tmp_path)
    cfg["telescope"]["adjust_data_bounds"] = {FILTER.lower(): [10, -10, 10, -10]}
    with pytest.raises(ValueError, match="reaches the data cutout's edge"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def grid_of(n, fov_arcsec, fov_y_arcsec=None):
    fov = [fov_arcsec, fov_y_arcsec or fov_arcsec] * u.arcsec  # (x, y)
    spatial = WcsAstropy(center=CENTER, shape=(n, n), fov=fov)
    return Grid(spatial=spatial, spectral=Color([3.6, 5.2] * u.um))


B1938_SKY = grid_of(512, 3.0)  # 5.86 mas: Nyquist 85.3, corner 120.7 cycles/arcsec


@pytest.mark.parametrize(
    "filter_name, pixel, subsample, grid, expected",
    [
        # f_s / 2 = 23.85, inside F277W's window 13.0 (6.5 m / 2.423 um) to 34.7
        ("F277W", 0.0629, 3, B1938_SKY, 3 / 0.0629 / 2),
        # f_s 143: the sky's corner 120.7 + 13.0 cannot alias, no band needed
        ("F277W", 0.0629, 9, B1938_SKY, None),
        # f_s / 2 = 48.4, inside F115W's window 31.1 (6.5 m / 1.013 um) to 65.7
        ("F115W", 0.031, 3, B1938_SKY, 3 / 0.031 / 2),
        # 0.0256" (x) by 0.0128" (y) sky pixels, Nyquist 19.5 and 39: f_s / 2 = 20
        # is capped below the lower one
        ("F277W", 0.075, 3, grid_of(117, 3.0, 1.5), 0.999 * 0.5 / (3.0 / 117)),
    ],
)
def test_automatic_band_limit(filter_name, pixel, subsample, grid, expected):
    from jubik.instruments.jwst.jwst_likelihoods import _band_limited
    from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
        NufftConfig,
    )

    auto = NufftConfig(band_limit="auto")
    band = _band_limited(
        auto, filter_name, subsample, pixel * u.arcsec, grid
    ).band_limit
    if expected is None:
        assert band is None
    else:
        assert band == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "filter_name, pixel, subsample, band, window",
    [
        # f_s 15.9 < 2 x 8.12 (6.5 m / 3.881 um): no window at all
        ("F444W", 0.063, 1, "auto", "8.12 and 7.753"),
        ("F277W", 0.0629, 3, 10.0, "13.01 and 34.69"),  # below k_c: cuts the passband
        ("F277W", 0.0629, 3, 40.0, "13.01 and 34.69"),  # above f_s - k_c: aliases
    ],
)
def test_band_limit_outside_the_window_raises(
    filter_name, pixel, subsample, band, window
):
    from jubik.instruments.jwst.jwst_likelihoods import _band_limited
    from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
        NufftConfig,
    )

    with pytest.raises(ValueError, match=f"optics' cutoff D/lambda {window}"):
        _band_limited(
            NufftConfig(band_limit=band),
            filter_name,
            subsample,
            pixel * u.arcsec,
            B1938_SKY,
        )


def test_band_limit_config_values():
    from jubik.instruments.jwst.jwst_likelihoods import _band_limited
    from jubik.instruments.jwst.parse.rotation_and_shift.rotation_and_shift import (
        NufftConfig,
    )

    parse = NufftConfig.from_yaml_dict
    assert parse({}).band_limit == "auto"
    for off in (None, False, "off"):
        assert parse({"band_limit": off}).band_limit is None
    assert parse({"band_limit": 20}).band_limit == 20.0
    for bad in (0, -1.0, True, "20"):
        with pytest.raises(ValueError, match="band_limit"):
            parse({"band_limit": bad})
    # an explicit band inside the window and no band pass through unchanged
    for band in (20.0, None):
        cfg = NufftConfig(band_limit=band)
        out = _band_limited(cfg, "F277W", 3, 0.0629 * u.arcsec, B1938_SKY)
        assert out.band_limit == band


def test_builder_hands_the_nufft_the_automatic_band(
    patched_seams, monkeypatch, tmp_path
):
    pytest.importorskip("jax_finufft")
    import jubik.instruments.jwst.rotation_and_shift.rotation_and_shift as rs

    seen = []
    real = rs.build_nufft_rotation_and_shift

    def spy(*args, **kwargs):
        seen.append(kwargs.get("band_limit"))
        return real(*args, **kwargs)

    monkeypatch.setattr(rs, "build_nufft_rotation_and_shift", spy)
    # a 0.025" sky: its corner 28.3 + F444W's 8.1 exceeds f_s = 2 / 0.063 = 31.7, so
    # the band is f_s / 2 = 15.9 cycles/arcsec, 0.397 cycles per sky pixel
    grid, domain = grid_of(64, 1.6), {"sky": jft.ShapeWithDtype((1, 64, 64), float)}
    for band, expected in (("absent", 2 / 0.063 / 2 * 0.025), ("off", None)):
        cfg = make_config(tmp_path)
        cfg["telescope"]["rotation_and_shift"] = {
            "nufft": {"mode": "constant"}
            | ({} if band == "absent" else {"band_limit": band}),
            "correction_priors": cfg["telescope"]["rotation_and_shift"][
                "correction_priors"
            ],
        }
        build_jwst_likelihoods(cfg, grid, domain)
        if expected is None:
            assert seen[-1] is None
        else:
            assert seen[-1] == pytest.approx((expected, expected), rel=1e-9)


class RotatedDitheredJwstData(FakeJwstData):
    """`FakeJwstData` turned 30 deg against the sky grid, each file `fake_<i>`
    dithered by its own fraction of a pixel."""

    def __init__(self, filepath: str):
        super().__init__(filepath)
        ii = int(Path(filepath).stem.split("_")[-1])
        self.wcs = WcsAstropy(
            center=CENTER.spherical_offsets_by(
                (0.017 + 0.029 * ii) * u.arcsec, (0.023 - 0.011 * ii) * u.arcsec
            ),
            shape=DATA_SHAPE,
            fov=self.wcs.geometry.fov_xy,
            position_angle=30 * u.deg,
        )


@pytest.fixture
def rotated_loads(monkeypatch, patched_seams):
    """Rotated and dithered fake data; the target data of every `load_data`
    call, each with a copy of its mask as loaded."""
    from jubik.instruments.jwst import jwst_likelihoods

    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, RotatedDitheredJwstData)
    loaded, load_data = [], jwst_likelihoods.load_data

    def spy(*args, **kwargs):
        out = load_data(*args, **kwargs)
        loaded.append((out.target_data, out.target_data.mask.copy()))
        return out

    monkeypatch.setattr(jwst_likelihoods, "load_data", spy)
    return loaded


@pytest.mark.parametrize(
    "model, rotation, rotation_reach",
    [
        ("shift", ["delta", 0.0, 0.0], 0.0),
        # 3 sigma about the grid centre (the fake boresight) moves the box's
        # corners, half a diagonal out, by this much
        ("rshift", ["normal", 0.0, 0.5], np.radians(1.5) * np.hypot(0.8, 0.72)),
    ],
)
def test_data_inset_masks_the_pixels_near_the_sky_edge(
    rotated_loads, tmp_path, model, rotation, rotation_reach
):
    # non-square sky pixels, 0.1" in x and 0.12" in y
    grid = Grid(
        spatial=WcsAstropy(center=CENTER, shape=(16, 12), fov=[1.6, 1.44] * u.arcsec),
        spectral=make_grid().spectral,
    )
    cfg = make_config(tmp_path, n_files=3)
    priors = cfg["telescope"]["rotation_and_shift"]["correction_priors"]
    priors["model"] = model
    priors["default"] = {"shift": ["normal", 0.01, 0.02], "rotation": rotation}
    cfg["telescope"]["target"]["data_arcsec_inset"] = 0.2
    sky_domain = {"sky": jft.ShapeWithDtype((1, 12, 16), float)}
    target = build_jwst_likelihoods(cfg, grid, sky_domain).target
    builder = target.likelihoods[0].builder

    ((target_data, loaded_mask),) = rotated_loads
    # plus the priors' reach: the shift's |mean| + 3 sigma, and the rotation's
    inset = 0.2 + 0.01 + 3 * 0.02 + rotation_reach
    s = cfg["telescope"]["target"]["subsample"]
    inside = []
    for centers in target_data.subsample_centers:
        # depth inside the box from the tangent-plane offsets to the grid centre
        east, north = CENTER.spherical_offsets_to(centers)
        depth = np.minimum(0.8 - np.abs(east.arcsec), 0.72 - np.abs(north.arcsec))
        h, w = depth.shape
        inside.append(depth.reshape(h // s, s, w // s, s).min(axis=(1, 3)) >= inset)
    expected = loaded_mask & np.stack(inside)
    assert 0 < expected.sum() < loaded_mask.sum()
    np.testing.assert_array_equal(builder.mask, expected)


@pytest.mark.parametrize("inset", ["absent", None])
def test_data_inset_off_keeps_the_loaded_mask(rotated_loads, tmp_path, inset):
    cfg = make_config(tmp_path, n_files=2)
    if inset != "absent":
        cfg["telescope"]["target"]["data_arcsec_inset"] = inset
    target = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN).target
    builder = target.likelihoods[0].builder
    ((_, loaded_mask),) = rotated_loads
    np.testing.assert_array_equal(builder.mask, loaded_mask)


def test_data_inset_leaves_the_model_on_kept_pixels(
    rotated_loads, monkeypatch, tmp_path
):
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", broad_build_webb_psf)

    def build(inset):
        cfg = make_config(tmp_path, n_files=2, zero_flux=True)
        priors = cfg["telescope"]["rotation_and_shift"]["correction_priors"]
        priors["default"]["shift"] = ["normal", 0.0, 0.02]
        if inset is not None:
            cfg["telescope"]["target"]["data_arcsec_inset"] = inset
        target = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN).target
        return target.likelihoods[0].builder

    full, inset = build(None), build(0.2)
    assert 0 < inset.mask.sum() < full.mask.sum()
    x = jft.random_like(jax.random.PRNGKey(2), full.response.domain)
    model = np.zeros(full.mask.shape)
    model[full.mask] = full.response(x)
    np.testing.assert_allclose(inset.response(x), model[inset.mask], rtol=1e-12)


@pytest.mark.parametrize(
    "inset, match",
    [
        (-0.1, "data_arcsec_inset must be >= 0"),
        # with the 0.3" shift reach, 0.5" is past the 1.6" box's centre
        (0.5, "masks every data pixel"),
    ],
)
def test_data_inset_refuses_negative_and_empty(patched_seams, tmp_path, inset, match):
    cfg = make_config(tmp_path)
    cfg["telescope"]["target"]["data_arcsec_inset"] = inset
    with pytest.raises(ValueError, match=match):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


CALIBRATION = f"{FILTER.lower()}_target_calibration"


def calibrated_and_plain(tmp_path, calibration):
    """The response of two exposures with `calibration` at random latents, and the
    response without it at the same latents, split per exposure."""
    plain = build_jwst_likelihoods(
        make_config(tmp_path, n_files=2), make_grid(), SKY_DOMAIN
    )
    cfg = make_config(tmp_path, n_files=2, calibration=calibration)
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)
    single = products.target.likelihoods[0]
    x = jft.random_like(jax.random.PRNGKey(1), single.response.domain)
    rest = {k: v for k, v in x.items() if k != CALIBRATION}
    counts = single.builder.mask.reshape(2, -1).sum(1)
    split = np.cumsum(counts)[:-1]
    return (
        products,
        plain,
        np.asarray(x[CALIBRATION]),
        np.split(np.asarray(single.response(x)), split),
        np.split(np.asarray(plain.target.likelihoods[0].response(rest)), split),
    )


def relative_factors(xi, mean, std):
    """A lognormal(mean, std) prior's factors over their geometric mean:
    exp(sigma (xi - mean xi)), sigma the log-space std; `mean` drops out."""
    sigma = np.sqrt(np.log1p((std / mean) ** 2))
    return np.exp(sigma * (xi - xi.mean()))


def test_calibration_gives_each_exposure_its_own_factor(patched_seams, tmp_path):
    products, plain, xi, calibrated, uncalibrated = calibrated_and_plain(
        tmp_path, {"default": ["lognormal", 1.0, 0.5]}
    )
    assert not any("calibration" in k for k in plain.target.likelihood.domain.tree)
    domain = products.target.likelihood.domain.tree
    assert set(domain) - set(plain.target.likelihood.domain.tree) == {CALIBRATION}
    assert domain[CALIBRATION].shape == (2, 1, 1)
    factors = relative_factors(xi.ravel(), 1.0, 0.5)
    assert not np.isclose(factors[0], factors[1])
    np.testing.assert_allclose(np.prod(factors), 1.0, rtol=1e-12)  # geometric mean 1
    for factor, scaled, model in zip(factors, calibrated, uncalibrated):
        np.testing.assert_allclose(scaled, factor * model, rtol=1e-12)
    assert np.isfinite(evaluate(products.target.likelihood))


def test_calibration_filter_key_overrides_default(patched_seams, tmp_path):
    _, _, xi, calibrated, uncalibrated = calibrated_and_plain(
        tmp_path,
        {"default": ["lognormal", 1.0, 0.5], FILTER.lower(): ["lognormal", 2.0, 0.2]},
    )
    factors = relative_factors(xi.ravel(), 2.0, 0.2)
    for factor, scaled, model in zip(factors, calibrated, uncalibrated):
        np.testing.assert_allclose(scaled, factor * model, rtol=1e-12)


@pytest.mark.parametrize(
    "prior",
    [["normal", 1.0, 0.1], ["lognormal", 1.0, 0.1, "log"]],
    ids=["normal", "log"],
)
def test_calibration_needs_a_plain_lognormal_prior(patched_seams, tmp_path, prior):
    # the geometric mean needs positive factors: a transformation may break that
    cfg = make_config(tmp_path, n_files=2, calibration={"default": prior})
    with pytest.raises(
        ValueError, match="need a lognormal prior without a transformation"
    ):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def test_calibration_key_of_no_filter_raises(patched_seams, tmp_path):
    cfg = make_config(
        tmp_path,
        calibration={"default": ["normal", 1.0, 0.5], "f277w": ["normal", 1.0, 0.1]},
    )
    with pytest.raises(ValueError, match=r"calibration: \['f277w'\] match no filter"):
        build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)
