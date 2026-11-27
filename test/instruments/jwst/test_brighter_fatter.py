"""The brighter-fatter charge migration: the operator, ``psf.brighter_fatter`` in
the target response, and `fit_psf_correction`'s ``brighter_fatter``."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u

from jubik.instruments.jwst.integration.brighter_fatter import charge_migration
from jubik.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik.instruments.jwst.jwst_response import JwstResponse
from jubik.instruments.jwst.masking.build_mask import build_mask
from jubik.instruments.jwst.parse.jwst_psf import JwstPsfKernelConfig
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.instruments.jwst.psf.psf_correction import (
    PsfCorrection,
    StarStamp,
    _StarModel,
    fit_psf_correction,
)

from test_jwst_likelihoods import (
    FILTER,
    JWST_DATA_SITES,
    SKY_DOMAIN,
    FakeJwstData,
    make_config,
    make_grid,
)
from test_jwst_psf_broadening import LOGNORMAL, wide_build_webb_psf
from test_psf_correction import SIZE, SUB, blank_stamp, kernel, zeros

BAND = FILTER.lower()
KAPPA = 9.02e-7  # per DN/int, B1938's F277W stars
# EFFINTTM (s) and PHOTMJSR (MJy/sr per DN/s) of the fake files fake_0, fake_1
TIME, PHOTMJSR = (100.0, 150.0), (0.5, 0.4)
DN_PER_UNIT = np.divide(TIME, PHOTMJSR)

# The prototype the operator was validated with (sptLensing tmp/bfe_test, 2026-11-25),
# as written there: Phi = K (x) Q with K's rings (k00, k10, k11, k20).
OFFSETS = {
    1: ((0, 1), (0, -1), (1, 0), (-1, 0)),
    2: ((1, 1), (1, -1), (-1, 1), (-1, -1)),
    3: ((0, 2), (0, -2), (2, 0), (-2, 0)),
}


def bfe(q, k):
    k = tuple(k) + (0.0,) * (4 - len(k))
    ny, nx = q.shape[-2:]
    padded = jnp.pad(q, [(0, 0)] * (q.ndim - 2) + [(2, 2), (2, 2)])
    phi = k[0] * q
    for ring, offsets in OFFSETS.items():
        for dy, dx in offsets:
            phi = (
                phi + k[ring] * padded[..., 2 + dy : 2 + dy + ny, 2 + dx : 2 + dx + nx]
            )
    fx = 0.5 * (q[..., :, :-1] + q[..., :, 1:]) * (phi[..., :, :-1] - phi[..., :, 1:])
    fy = 0.5 * (q[..., :-1, :] + q[..., 1:, :]) * (phi[..., :-1, :] - phi[..., 1:, :])
    zx, zy = jnp.zeros_like(q[..., :, :1]), jnp.zeros_like(q[..., :1, :])
    out = jnp.concatenate([fx, zx], -1) - jnp.concatenate([zx, fx], -1)
    out = out + jnp.concatenate([fy, zy], -2) - jnp.concatenate([zy, fy], -2)
    return q - out


def charges(seed=1):
    """Two exposures of 9 x 11 pixels, -1000 to 30 000 DN/int with exact zeros (a
    background-subtracted model; the sky outside the grid)."""
    q = np.random.default_rng(seed).uniform(-1e3, 3e4, (2, 9, 11))
    q[:, ::3, ::4] = 0.0
    return q


# --- the operator -------------------------------------------------------------------


def test_conserves_the_charge():
    q = charges()
    moved = np.asarray(charge_migration(q, np.array([KAPPA, 3 * KAPPA])[:, None, None]))
    assert np.abs(moved - q).max() > 1e-3 * q.max()  # it moves charge
    np.testing.assert_allclose(moved.sum((-2, -1)), q.sum((-2, -1)), rtol=1e-13)


def test_kappa_zero_is_the_identity():
    q = charges()
    np.testing.assert_array_equal(charge_migration(q, 0.0), q)


def test_matches_the_prototype():
    q = charges(2)
    kappa = np.array([KAPPA, 2.5 * KAPPA])
    expected = np.stack([bfe(q[e], (kappa[e],)) for e in range(2)])
    moved = charge_migration(q, kappa[:, None, None])
    np.testing.assert_allclose(moved, expected, rtol=1e-12, atol=1e-12 * q.max())
    # a point source loses charge to its four neighbours: it broadens
    star = np.zeros((7, 7))
    star[3, 3] = 2e4
    np.testing.assert_allclose(charge_migration(star, KAPPA), bfe(star, (KAPPA,)))


def test_gradient_is_finite():
    q = charges(3)
    grad = jax.grad(lambda c: jnp.sum(charge_migration(c, KAPPA) ** 2))(q)
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(  # d/dQ_p sum Q'^2 at kappa = 0 is 2 Q_p
        jax.grad(lambda c: jnp.sum(charge_migration(c, 0.0) ** 2))(q), 2 * q
    )


# --- psf.brighter_fatter in the target response -------------------------------------


class ExposureJwstData(FakeJwstData):
    """`FakeJwstData` with EFFINTTM and PHOTMJSR, per file ``fake_<i>``."""

    unit = u.MJy / u.sr
    time, photmjsr = TIME, PHOTMJSR

    def __init__(self, filepath):
        super().__init__(filepath)
        i = int(Path(filepath).stem.split("_")[-1])
        self.dm.meta.exposure = SimpleNamespace(integration_time=self.time[i])
        self.dm.meta.photometry = SimpleNamespace(
            conversion_megajanskys=self.photmjsr[i]
        )
        self.meta.unit = self.unit


@pytest.fixture
def exposures(monkeypatch):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, ExposureJwstData)
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", wide_build_webb_psf)


def build(tmp_path, psf=(), **telescope):
    """The filter's likelihood products for two exposures."""
    tmp_path.mkdir(exist_ok=True)
    cfg = make_config(tmp_path, n_files=2)
    cfg["telescope"]["psf"] |= dict(psf)
    cfg["telescope"] |= telescope
    return build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)


def latents(products, seed=3):
    x = jft.random_like(jax.random.PRNGKey(seed), products.target.likelihood.domain)
    x = dict(x.tree if isinstance(x, jft.Vector) else x)
    x["sky"] = 1e3 * np.abs(np.asarray(x["sky"]))
    return x


def unmasked(response, x):
    """The response's image before its mask, (n_exposures, ny, nx)."""
    r = response
    return JwstResponse(
        r.sky_model,
        r.psf,
        r.unit_conversion,
        r.integrate,
        r.zero_flux_model,
        build_mask(None),
        r.calibration_model,
    )(x)


def test_composes_with_correction_box_calibration_and_broadening(exposures, tmp_path):
    """The charge of each exposure's whole model image, in DN/int by its own
    EFFINTTM / PHOTMJSR, moves as the prototype moves it; the correction, the
    sub-pixel box, the calibration, the broadening and the zero flux come first."""
    texture = 0.1 * np.random.default_rng(0).normal(size=(5, 5))
    rest = {
        "calibration": {"default": ["lognormal", 1.0, 0.5]},
        "psf_broadening": {"default": LOGNORMAL},
        "zero_flux": {"default": ["normal", 0.5, 0.1]},
    }

    def correction(kappa):
        path = tmp_path / f"c{kappa}.npz"
        PsfCorrection(
            BAND, 2, texture, 0.5, pixel_box=True, brighter_fatter=kappa
        ).save(path)
        return {"pixel_box": True, "correction": {BAND: str(path)}}

    plain = build(tmp_path / "plain", correction(0.0), **rest)
    (single,) = plain.target.likelihoods
    assert single.response.brighter_fatter is None
    x = latents(plain)
    image = np.asarray(unmasked(single.response, x | plain.target.sky_filter(x)))
    # a kappa that moves a few percent of the brightest pixel's charge
    kappa = 0.05 / (DN_PER_UNIT.max() * np.abs(image).max())
    psf = correction(kappa) | {"brighter_fatter": {FILTER: kappa}}
    products = build(tmp_path / "bfe", psf, **rest)
    (moved,) = products.target.likelihoods
    assert set(moved.response.domain) == set(single.response.domain)
    np.testing.assert_allclose(
        moved.response.brighter_fatter.ravel(), kappa * DN_PER_UNIT
    )

    dn = image * DN_PER_UNIT[:, None, None]
    expected = np.stack([bfe(dn[e], (kappa,)) for e in range(2)])
    expected = expected / DN_PER_UNIT[:, None, None]
    mask = single.builder.mask
    out = np.asarray(moved.response(x | products.target.sky_filter(x)))
    np.testing.assert_allclose(
        out, expected[mask], rtol=1e-10, atol=1e-12 * np.abs(image).max()
    )
    assert np.abs(out - image[mask]).max() > 1e-3 * np.abs(image).max()

    grad = jax.grad(products.target.likelihood)(x)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grad))


def test_needs_data_in_mjy_per_sr_with_photmjsr(exposures, monkeypatch, tmp_path):
    psf = {"brighter_fatter": {BAND: KAPPA}}
    bad = {"photmjsr": (0.5, None), "time": (100.0, None), "unit": u.DN / u.s}
    for key, value in bad.items():
        with monkeypatch.context() as m:  # one bad value at a time
            m.setattr(ExposureJwstData, key, value)
            with pytest.raises(ValueError, match="needs data in MJy/sr with EFFINTTM"):
                build(tmp_path / key, psf)


@pytest.mark.parametrize(
    "kappa, match",
    [
        ({"f277w": KAPPA}, r"psf brighter_fatter: \['f277w'\] match no filter"),
        ({BAND: -KAPPA}, "must be finite and >= 0"),
        ({BAND: float("nan")}, "must be finite and >= 0"),
        ({BAND: float("inf")}, "must be finite and >= 0"),
        (KAPPA, "must map filter -> kappa"),
    ],
)
def test_bad_config_raises(exposures, tmp_path, kappa, match):
    with pytest.raises(ValueError, match=match):
        build(tmp_path, {"brighter_fatter": kappa})


def test_correction_needs_its_kappa(exposures, tmp_path):
    raw = {
        "webbpsf_path": "",
        "psf_library": str(tmp_path),
        "psf_arcsec_extension": 0.4,
    }
    data = ExposureJwstData("fake_0.fits")
    for learned, configured in ((KAPPA, None), (0.0, KAPPA), (KAPPA, 2 * KAPPA)):
        PsfCorrection(BAND, 2, np.zeros((1, 1)), 0.3, brighter_fatter=learned).save(
            tmp_path / "c.npz"
        )
        psf = raw | {"correction": {BAND: str(tmp_path / "c.npz")}}
        if configured is not None:
            psf["brighter_fatter"] = {BAND: configured}
        with pytest.raises(ValueError, match="learned with brighter_fatter"):
            jwst_kernel.load_psf_kernel(
                data, 2, data.wcs.center, JwstPsfKernelConfig.from_yaml_dict(psf)
            )
    psf["brighter_fatter"] = {BAND: KAPPA}
    PsfCorrection(BAND, 2, np.zeros((1, 1)), 0.3, brighter_fatter=KAPPA).save(
        tmp_path / "c.npz"
    )
    jwst_kernel.load_psf_kernel(
        data, 2, data.wcs.center, JwstPsfKernelConfig.from_yaml_dict(psf)
    )


# --- fit_psf_correction(brighter_fatter=...) ----------------------------------------


def stars_with_charge_migration(w0, kappa, n=30, dn_per_unit=2000.0, seed=0):
    """Point sources through a Gaussian of width `w0` kernel px, 1e4 to 1e5 DN/int,
    their charge moved by the prototype at `kappa`, in data units of 1/`dn_per_unit`
    DN/int; noiseless (read and shot noise only weight the pixels), so the fit's
    slope is the operator's, not the noise's (+-6e-7 with the noise drawn). The
    fluence of each is its peak in DN/int."""
    rng = np.random.default_rng(seed)
    base = kernel()
    model = _StarModel([blank_stamp(base)] * n, SUB)
    model.flux0 = 10 ** rng.uniform(4, 5, n)
    model.background0, model.noise = np.full(n, 5.0), np.ones(n)
    x = zeros(model) | {"star_position": rng.uniform(-1, 1, (n, 2))}
    static = np.asarray(model(x, None, w0))  # DN/int
    data = np.asarray(bfe(jnp.asarray(static), (kappa,))) / dn_per_unit
    std = np.sqrt(4.0 + static.clip(0) / 2)
    mask = np.ones((SIZE, SIZE), bool)
    peaks = (static - 5.0).max((1, 2))
    return [
        StarStamp(d, s / dn_per_unit, mask, base, p, dn_per_unit)
        for d, s, p in zip(data, std, peaks)
    ]


def test_fit_with_the_operator_keeps_what_charge_migration_does_not_explain():
    """Stars broadened by charge migration, fitted with the operator, give the
    width and slope (~0) of the same stars without it, fitted without; fitted
    without the operator, their width grows with the peak. At B1938's w0 of
    0.053 kernel px the width prior, not the stars, sets the fitted w0 (0.1 with
    a prior of 0.5 +- 0.5), so the reference is the same fit, not the truth."""
    w0 = 0.053
    stamps = stars_with_charge_migration(w0, KAPPA)
    options = dict(width=(0.5, 0.5), fluence=2e4, n_iterations=5)
    without, _ = fit_psf_correction(stamps, SUB, "f", **options)
    learned, _ = fit_psf_correction(stamps, SUB, "f", brighter_fatter=KAPPA, **options)
    plain = stars_with_charge_migration(w0, 0.0)
    reference, _ = fit_psf_correction(plain, SUB, "f", **options)
    assert without.width_slope > 2e-6  # kernel px² per DN/int
    # the reference's slope, -1.6e-7, is the width fit's at so small a w0
    diff = learned.width_slope - reference.width_slope
    assert abs(diff) < 0.02 * without.width_slope
    w = [c.at_fluence(0).width for c in (learned, reference)]
    assert np.isclose(*w, rtol=0.01)
    assert learned.brighter_fatter == KAPPA and without.brighter_fatter == 0.0
    # stamps in an unknown unit cannot take kappa per DN/int
    with pytest.raises(ValueError, match="needs every StarStamp.dn_per_unit"):
        fit_psf_correction([blank_stamp(kernel())], SUB, "f", brighter_fatter=KAPPA)
