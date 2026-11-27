"""``psf.joint_stars``: the PSF correction learned in the lens fit together with
the field's stars, which share its texture, width and kappa."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest

from jubik.instruments.jwst.parse.jwst_psf import JointStars, JwstPsfKernelConfig
from jubik.instruments.jwst.psf.psf_correction import (
    PsfCorrection,
    StarStamp,
    _StarModel,
    fit_psf_correction,
    joint_correction,
    load_star_stamps,
    log_texture_model,
    save_star_stamps,
    with_pixel_box,
)
from jubik.instruments.jwst.psf.psf_operator import PsfStatic

from test_brighter_fatter import (
    BAND,
    KAPPA,
    ExposureJwstData,
    bfe,
    build,
    exposures,  # noqa: F401
    latents,
    stars_with_charge_migration,
)
from test_jwst_likelihoods import JWST_DATA_SITES
from test_jwst_psf_broadening import wide_build_webb_psf
from test_psf_correction import SIZE, blank_stamp, kernel, zeros

SUB = 2  # make_config's target.subsample
RADIUS = 6  # kernel px, the texture's
WIDTH = (0.8, 0.5)  # kernel px, the width's log-normal prior
DN = 2000.0  # DN/int per data unit of the stamps
TEXTURE = f"{BAND}_psf_texture"


def base_kernel():
    """The fake STPSF kernel of every exposure and star."""
    return wide_build_webb_psf(None, None, None, None, SUB)


def write_stamps(path, n=8, texture=None, width=None, seed=0):
    """`n` stars, 1e4 to 1e5 DN/int, through the base kernel with a texture and
    width (values or None) and the box, with read and shot noise, in data units
    of 1 / DN DN/int."""
    rng = np.random.default_rng(seed)
    k = base_kernel()
    model = _StarModel([blank_stamp(k)] * n, SUB, pixel_box=True)
    model.flux0 = 10 ** rng.uniform(4, 5, n)
    model.background0, model.noise = np.full(n, 5.0), np.ones(n)
    x = zeros(model) | {"star_position": rng.uniform(-1, 1, (n, 2))}
    clean = np.asarray(model(x, texture, width))
    std = np.sqrt(4.0 + clean.clip(0) / 2)
    data = clean + std * rng.normal(size=clean.shape)
    mask = np.ones((SIZE, SIZE), bool)
    mask[0, :3] = False  # a masked corner
    stamps = [
        StarStamp(d / DN, s / DN, mask, k, f, DN)
        for d, s, f in zip(data, std, model.flux0)
    ]
    save_star_stamps(path, stamps, SUB, star=np.arange(n))
    return stamps


def joint_psf(path, kappa=(KAPPA, 0.5 * KAPPA), **options):
    entry = {"stamps": str(path), "radius": RADIUS, "width": list(WIDTH)} | options
    if kappa is not None:
        entry["brighter_fatter"] = list(kappa)
    return {"pixel_box": True, "joint_stars": {BAND: entry}}


def texture_and_width(x, orthogonal=False):
    """The texture and width a joint_psf config gives the latents `x`."""
    c = base_kernel().shape[0] // 2
    weight = base_kernel()[c - RADIUS : c + RADIUS + 1, c - RADIUS : c + RADIUS + 1]
    texture = log_texture_model(
        RADIUS, 0.1, -2.0, prefix=TEXTURE, weight=weight if orthogonal else None
    )
    width = jft.LogNormalPrior(*WIDTH, name=f"{BAND}_psf_width", shape=())
    return np.asarray(texture(x)), float(width(x))


def kappa_of(x):
    """Kappa of a joint_psf config's prior at the latents `x`."""
    prior = jft.LogNormalPrior(KAPPA, 0.5 * KAPPA, name="k", shape=())
    return prior({"k": x[f"{BAND}_psf_brighter_fatter"]})


def width_at_fluence(x):
    w1 = jft.LogNormalPrior(*WIDTH, name="w", shape=())
    return float(w1({"w": x[f"{BAND}_psf_width_at_fluence"]}))


def tree(domain):
    return domain.tree if isinstance(domain, jft.Vector) else domain


def energy(likelihood, x):
    return float(likelihood(jft.Vector(x)))


def test_stamps_file_round_trip(tmp_path):
    stamps = write_stamps(tmp_path / "s.npz", n=3)
    back, subsample = load_star_stamps(tmp_path / "s.npz")
    assert subsample == SUB and len(back) == 3
    for a, b in zip(stamps, back):
        for field in ("data", "std", "mask", "kernel"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        assert (a.fluence, a.dn_per_unit) == (b.fluence, b.dn_per_unit)
    with np.load(tmp_path / "s.npz") as f:  # the ids stay alongside
        np.testing.assert_array_equal(f["star"], np.arange(3))


class DeterministicJwstData(ExposureJwstData):
    """`ExposureJwstData` whose data does not depend on the process's str hash."""

    def __init__(self, filepath):
        super().__init__(filepath)
        nan = np.isnan(self.dm.data)
        seed = int(str(filepath).rsplit("_", 1)[-1].split(".")[0])
        self.dm.data = np.random.default_rng(seed).normal(size=nan.shape)
        self.dm.data[nan] = np.nan


def test_off_keeps_the_fixed_kernel_likelihood(exposures, monkeypatch, tmp_path):
    """Without joint_stars the likelihood is the fixed kernel's: its energy and
    gradient at fixed latents are those of the code before joint_stars
    (jwst-charge-migration, aa2e1635), with a correction file, the box, kappa,
    calibration and zero flux all on."""
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, DeterministicJwstData)
    texture = 0.1 * np.random.default_rng(0).normal(size=(5, 5))
    path = tmp_path / "c.npz"
    PsfCorrection(BAND, SUB, texture, 0.5, pixel_box=True, brighter_fatter=KAPPA).save(
        path
    )
    psf = {
        "pixel_box": True,
        "correction": {BAND: str(path)},
        "brighter_fatter": {BAND: KAPPA},
    }
    products = build(
        tmp_path / "b",
        psf,
        calibration={"default": ["lognormal", 1.0, 0.5]},
        zero_flux={"default": ["normal", 0.5, 0.1]},
    )
    (single,) = products.target.likelihoods
    assert (
        isinstance(single.response.psf, PsfStatic) and products.target.joint_stars == []
    )
    x = latents(products)
    likelihood = products.target.likelihood
    grad = jax.grad(lambda y: likelihood(jft.Vector(y)))(x)
    # the values at aa2e1635, the same test
    assert np.isclose(energy(likelihood, x), 369615559.2233298, rtol=1e-13)
    norm = np.sqrt(sum(float(jnp.sum(g**2)) for g in jax.tree.leaves(grad)))
    assert np.isclose(norm, 13415766.129993439, rtol=1e-12)


def test_star_term_is_the_star_fits_model(exposures, tmp_path):
    """The star term is `fit_psf_correction`'s likelihood with the latents named
    by the filter; with a learned kappa, the stars' charge moves by it."""
    stamps = write_stamps(tmp_path / "s.npz")
    data = np.array([s.data for s in stamps])
    std = np.array([s.std for s in stamps])
    mask = np.array([s.mask for s in stamps])

    def chi2(models):
        return 0.5 * float(np.sum(((data - models) / std)[mask] ** 2))

    # fixed kappa, at the star fit's starting point
    fixed = build(
        tmp_path / "fixed",
        joint_psf(tmp_path / "s.npz", kappa=None) | {"brighter_fatter": {BAND: KAPPA}},
    )
    (stars,) = fixed.target.joint_stars
    _, start = fit_psf_correction(
        stamps,
        SUB,
        BAND,
        radius=RADIUS,
        width=WIDTH,
        pixel_box=True,
        brighter_fatter=KAPPA,
        n_iterations=0,
    )
    x0 = {k: np.zeros(v.shape) for k, v in tree(stars.domain).items()}
    assert np.isclose(energy(stars, x0), chi2(start), rtol=1e-10)

    # learned kappa, at random latents: the stars' static model, then the prototype
    products = build(tmp_path / "learned", joint_psf(tmp_path / "s.npz"))
    (stars,) = products.target.joint_stars
    x = latents(products, seed=5)
    assert {f"{BAND}_star_flux", f"{BAND}_psf_brighter_fatter"} <= set(x)
    texture, width = texture_and_width(x)
    kappa = kappa_of(x)
    static = _StarModel(stamps, SUB, pixel_box=True)
    plain = {k.removeprefix(f"{BAND}_"): v for k, v in x.items()}
    dn = np.asarray(static(plain, texture, width)) * DN
    models = np.stack([bfe(d, (float(kappa),)) for d in dn]) / DN
    assert np.abs(models - dn / DN).max() > 1e-3 * np.abs(dn / DN).max()
    assert np.isclose(energy(stars, x), chi2(models), rtol=1e-10)


def test_lens_and_stars_share_the_correction(exposures, tmp_path):
    """At any latents, the lens's response is the one with the correction
    file of the latents' texture, width and kappa; the texture's gradient is the
    lens's plus the stars'."""
    write_stamps(tmp_path / "s.npz")
    products = build(tmp_path / "joint", joint_psf(tmp_path / "s.npz"))
    (single,) = products.target.likelihoods
    x = latents(products)
    texture, width = texture_and_width(x)
    kappa = float(kappa_of(x))

    kernels = np.asarray(single.response.psf.kernel(x))
    corrected = PsfCorrection(BAND, SUB, texture, width).apply(base_kernel())
    np.testing.assert_allclose(
        kernels, np.stack([with_pixel_box(corrected)] * 2), atol=1e-15
    )

    path = tmp_path / "c.npz"
    PsfCorrection(
        BAND, SUB, texture, width, pixel_box=True, brighter_fatter=kappa
    ).save(path)
    psf = {
        "pixel_box": True,
        "correction": {BAND: str(path)},
        "brighter_fatter": {BAND: kappa},
    }
    fixed = build(tmp_path / "fixed", psf)
    (frozen,) = fixed.target.likelihoods
    sky = products.target.sky_filter(x)
    out = np.asarray(single.response(x | sky))
    np.testing.assert_allclose(
        out, frozen.response(x | sky), rtol=1e-10, atol=1e-12 * np.abs(out).max()
    )

    lens = replace(products.target, joint_stars=[]).likelihood
    (stars,) = products.target.joint_stars
    parts = [
        jax.grad(lambda y, lh=lh: lh(jft.Vector(y)))(x)
        for lh in (products.target.likelihood, lens, stars)
    ]
    # the texture's excitations; its spectrum's latents are fixed, do nothing
    keys = [f"{TEXTURE}xi", f"{BAND}_psf_width", f"{BAND}_psf_brighter_fatter"]
    assert set(keys) <= set(tree(stars.domain))
    for key in keys:
        total, from_lens, from_stars = (np.asarray(p[key]) for p in parts)
        assert np.any(from_lens != 0) and np.any(from_stars != 0)
        np.testing.assert_allclose(total, from_lens + from_stars, rtol=1e-10)


def test_lens_gets_the_orthogonal_texture_and_the_width_at_the_fluence(
    exposures, tmp_path
):
    write_stamps(tmp_path / "s.npz")
    psf = joint_psf(tmp_path / "s.npz", orthogonal=True, fluence=3e4)
    products = build(tmp_path, psf)
    (single,) = products.target.likelihoods
    x = latents(products)
    texture, w0 = texture_and_width(x, orthogonal=True)
    w1 = width_at_fluence(x)
    assert abs(w1 - w0) > 0.1
    kernels = np.asarray(single.response.psf.kernel(x))
    corrected = PsfCorrection(BAND, SUB, texture, w1).apply(base_kernel())
    np.testing.assert_allclose(
        kernels, np.stack([with_pixel_box(corrected)] * 2), atol=1e-15
    )


def test_gradient_and_metric_are_finite(exposures, tmp_path):
    """Everything on, the width at the fluence so far below w0 that the
    brightest stars' variance law crosses zero (and is floored)."""
    stamps = write_stamps(tmp_path / "s.npz")
    psf = joint_psf(tmp_path / "s.npz", orthogonal=True, fluence=3e4)
    products = build(tmp_path, psf, zero_flux={"default": ["normal", 0.5, 0.1]})
    likelihood = products.target.likelihood
    x = latents(products)
    assert f"{BAND}_psf_width_at_fluence" in x
    x[f"{BAND}_psf_width"], x[f"{BAND}_psf_width_at_fluence"] = 2.0, -3.0
    w0, w1 = texture_and_width(x)[1], width_at_fluence(x)
    ratio = max(s.fluence for s in stamps) / 3e4
    assert w0**2 + (w1**2 - w0**2) * ratio < 0
    x = jft.Vector(x)
    grad = jax.grad(likelihood)(x)
    metric = likelihood.metric(x, x)
    for tree in (grad, metric):
        assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(tree))


def joint(**entry):
    return {"joint_stars": {BAND: {"stamps": "s.npz"} | entry}}


@pytest.mark.parametrize(
    "psf, match",
    [
        (joint(radious=3), "unexpected keyword argument 'radious'"),
        ({"joint_stars": {BAND: "s.npz"}}, "psf joint_stars f444w"),
        (joint(brighter_fatter=[KAPPA]), "log-normal"),
        (joint(brighter_fatter=[KAPPA, -1.0]), "log-normal"),
        (joint(fluence="2.2e4x"), "could not convert string to float"),
        (joint(radius=0), "radius must be >= 1; omit it for no texture"),
        (joint(brighter_fatter=[KAPPA, 1.0]) | {"brighter_fatter": {BAND: 1.0}}, "one"),
        (joint() | {"correction": {BAND: "c.npz"}}, "drop its psf correction"),
        ({"joint_stars": {"f277w": {"stamps": "s.npz"}}}, "match no filter"),
    ],
)
def test_bad_config_raises(exposures, tmp_path, psf, match):
    with pytest.raises(ValueError, match=match):
        build(tmp_path, psf)


def test_numbers_yaml_reads_as_strings_are_numbers():
    """PyYAML reads 1e-6, without a dot, as a string."""
    raw = {"webbpsf_path": "", "psf_library": "", "psf_arcsec_extension": 0.4}
    entry = {"stamps": "s.npz", "width": ["4e-1", 0.3], "fluence": "2.2e4"}
    entry |= {"brighter_fatter": ["9.02e-7", "2e-7"], "fluctuations": "1e-1"}
    config = JwstPsfKernelConfig.from_yaml_dict(raw | {"joint_stars": {BAND: entry}})
    assert config.joint_stars[BAND] == JointStars(
        "s.npz", None, 0.1, -2.0, (0.4, 0.3), 2.2e4, False, (9.02e-7, 2e-7)
    )


def test_conflicting_data_raise(exposures, tmp_path):
    write_stamps(tmp_path / "s.npz")
    broadening = {"psf_broadening": {"default": ["lognormal", 1e-3, 1e-3]}}
    with pytest.raises(ValueError, match="psf_broadening and psf joint_stars both"):
        build(tmp_path / "a", joint_psf(tmp_path / "s.npz"), **broadening)
    save_star_stamps(tmp_path / "s3.npz", load_star_stamps(tmp_path / "s.npz")[0], 3)
    with pytest.raises(ValueError, match="at subsample 3, the target's at 2"):
        build(tmp_path / "b", joint_psf(tmp_path / "s3.npz"))


def test_gaia_alignment_raises(exposures, tmp_path):
    from test_jwst_likelihoods import make_config

    from jubik.instruments.jwst.parse.parsing_step import ConfigParserJwst

    cfg = make_config(tmp_path, gaia=True)
    cfg["telescope"]["psf"] |= {"joint_stars": {BAND: {"stamps": "s.npz"}}}
    with pytest.raises(ValueError, match="gaia_alignment stars cannot"):
        ConfigParserJwst.from_yaml_dict(cfg)


def fit(likelihood, iterations=6):
    position = jft.Vector(
        jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), tree(likelihood.domain))
    )
    quiet = dict(name=None, cg_kwargs=dict(name=None))
    samples, _ = jft.optimize_kl(
        likelihood,
        position,
        key=jax.random.PRNGKey(0),
        n_total_iterations=iterations,
        n_samples=0,
        draw_linear_kwargs=dict(cg_name=None, cg_kwargs={}),
        nonlinearly_update_kwargs=dict(minimize_kwargs=quiet),
        kl_kwargs=dict(minimize_kwargs=quiet | dict(maxiter=20)),
    )
    return samples.pos.tree


def test_kappa_is_recovered_on_stars_with_a_known_kappa():
    """Noiseless stars whose charge moved at kappa, fitted with the width and
    kappa learned, kappa's prior centred twice too high."""
    stamps = stars_with_charge_migration(0.5, KAPPA)
    prior = jft.LogNormalPrior(2 * KAPPA, 2 * KAPPA, name="kappa", shape=())
    sky = jft.ShapeWithDtype((1, 8, 8), float)
    _, stars = joint_correction(
        stamps, 3, kernel()[None], sky, width=(0.5, 0.5), brighter_fatter=prior
    )
    x = fit(stars)
    assert np.isclose(prior(x), KAPPA, rtol=0.03)


def test_joint_fit_recovers_an_injected_correction(exposures, tmp_path):
    """Lens data made with a correction file's kernel at 3% noise and stars made
    with the same texture and width, the sky held at its truth: the joint fit
    gets the kernel ten times closer to the truth than the uncorrected one and
    the width within 2%, and fits the lens and the stars at the noise; the lens
    alone misses both."""
    write_stamps(tmp_path / "s.npz")  # replaced below, once the truth is drawn
    psf = joint_psf(tmp_path / "s.npz", kappa=None, orthogonal=True)
    products = build(tmp_path / "a", psf)
    truth = latents(products, seed=11)
    truth[f"{BAND}_psf_width"] = np.array(1.0)
    texture, width = texture_and_width(truth, orthogonal=True)
    assert 0.05 < np.abs(texture).max() < 0.5 and 1.2 < width < 1.25
    stamps = write_stamps(
        tmp_path / "s.npz", n=16, texture=texture, width=width, seed=1
    )
    products = build(tmp_path / "b", psf)

    correction = PsfCorrection(BAND, SUB, texture, width, pixel_box=True)
    correction.save(tmp_path / "c.npz")
    fixed = build(
        tmp_path / "c",
        {"pixel_box": True, "correction": {BAND: str(tmp_path / "c.npz")}},
    )
    sky = products.target.sky_filter(truth)
    image = np.asarray(fixed.target.likelihoods[0].response(truth | sky))
    (single,) = products.target.likelihoods
    noise = 3e-2 * np.abs(image).max()
    rng = np.random.default_rng(2)
    data = single.builder.data.copy()
    data[single.builder.mask] = image + noise * rng.normal(size=image.shape)
    single.builder.data, single.builder.std = data, np.full(data.shape, noise)

    def fit_with_the_sky_held(likelihood):
        domain = {k: v for k, v in tree(likelihood.domain).items() if k != "sky"}
        held = jft.Model(
            lambda x: jft.Vector(tree(x) | {"sky": truth["sky"]}), domain=domain
        )
        return fit(likelihood.amend(held, domain=jft.Vector(domain)))

    true_kernel = with_pixel_box(correction.apply(base_kernel()))
    untouched = np.abs(with_pixel_box(base_kernel()) - true_kernel).sum()

    def errors(learned):
        kernel = np.asarray(single.response.psf.kernel(learned))
        return (
            np.abs(kernel - true_kernel).sum() / untouched,
            abs(texture_and_width(learned)[1] / width - 1),
        )

    lens = replace(products.target, joint_stars=[]).likelihood
    learned = fit_with_the_sky_held(products.target.likelihood)
    assert max(errors(learned)) < 0.1 and errors(learned)[1] < 0.02  # 0.024, 0.001
    (stars,) = products.target.joint_stars
    chi2 = [
        2 * energy(stars, learned) / sum(int(s.mask.sum()) for s in stamps),
        2 * energy(lens, learned | {"sky": truth["sky"]}) / single.builder.mask.sum(),
    ]
    assert max(chi2) < 1.15  # the reduced chi2 std is ~0.04
    alone = errors(fit_with_the_sky_held(lens))
    assert alone[0] > 0.1 and alone[1] > 0.02  # 0.16, 0.03
