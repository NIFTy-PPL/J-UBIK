"""The learned kernel correction `G_w * N[K exp(t)]` and its star fit."""

import numpy as np
import pytest

from jubik.instruments.jwst.parse.jwst_psf import JwstPsfKernelConfig
from jubik.instruments.jwst.parse.parsing_step import ConfigParserJwst
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.instruments.jwst.psf.psf_correction import (
    PsfCorrection,
    StarStamp,
    _StarModel,
    fit_psf_correction,
)
from jubik.instruments.jwst.psf.psf_operator import _center_even_axes

from test_jwst_likelihoods import (
    FakeJwstData,
    fake_build_webb_psf,
    make_config,
    patched_seams,  # noqa: F401
)

RNG = np.random.default_rng(1938)
SUB, SIZE = 3, 11
NO_TEXTURE = np.zeros((1, 1))


def kernel(k=41, sigma=2.0, ring=6.0):
    y, x = np.mgrid[:k, :k] - (k - 1) / 2
    r = np.hypot(x, y)
    core = np.exp(-0.5 * (r / sigma) ** 2) + 0.03 * np.exp(-0.5 * (r - ring) ** 2)
    g = core + 1e-3 * np.exp(-r / 3)  # a wing fading towards the border, as STPSF's
    return g / g.sum()


def texture(radius, amplitude=0.15):
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r = np.hypot(x, y)
    t = amplitude * np.exp(-0.5 * ((x - 2) ** 2 + (y + 1) ** 2) / 2.5**2)
    return t * np.cos(np.pi * np.clip(r / radius - 0.5, 0, 0.5)) ** 2


def second_moment(k):
    y, x = np.indices(k.shape) - (np.array(k.shape)[:, None, None] - 1) / 2
    return float((k * (x**2 + y**2)).sum() / k.sum())


def blank_stamp(k):
    ones = np.ones((SIZE, SIZE))
    return StarStamp(ones, ones, ones.astype(bool), k)


def zeros(model):
    return {key: np.zeros(v.shape) for key, v in model.domain.items()}


@pytest.mark.parametrize("k", [40, 41])
def test_identity_and_flux(k):
    base = kernel(k)
    identity = PsfCorrection("f277w", SUB, NO_TEXTURE, 0.0).apply(base)
    np.testing.assert_allclose(identity, _center_even_axes(base), atol=1e-15)
    t = RNG.normal(0, 0.3, (15, 15))
    corrected = PsfCorrection("f277w", SUB, t, 0.7).apply(base)
    assert corrected.shape == (41, 41)  # odd, centred
    np.testing.assert_allclose(corrected.sum(), base.sum(), rtol=1e-12)


def test_texture_scales_the_middle_and_width_adds_variance():
    base = kernel()
    t = np.zeros((5, 5))
    t[2, 2] = np.log(2.0)
    ratio = PsfCorrection("f", SUB, t, 0.0).apply(base) / base
    c = base.shape[0] // 2
    assert np.isclose(ratio[c, c] / ratio[0, 0], 2.0)  # only the middle pixel doubles
    broadened = PsfCorrection("f", SUB, NO_TEXTURE, 1.3).apply(base)
    np.testing.assert_allclose(
        second_moment(broadened) - second_moment(base), 2 * 1.3**2, rtol=1e-2
    )


def test_texture_larger_than_kernel_raises():
    with pytest.raises(ValueError, match="larger than the kernel"):
        PsfCorrection("f", SUB, np.zeros((43, 43)), 0.0).apply(kernel(41))


def test_save_load_round_trip(tmp_path):
    c = PsfCorrection("f115w", SUB, RNG.normal(size=(7, 7)), 0.4, 2e4, 3e-6)
    c.save(tmp_path / "c.npz")
    d = PsfCorrection.load(tmp_path / "c.npz")
    fields = ("filter", "subsample", "width", "fluence", "width_slope")
    assert [getattr(d, f) for f in fields] == [getattr(c, f) for f in fields]
    np.testing.assert_array_equal(d.log_texture, c.log_texture)


def test_at_fluence_moves_the_variance_linearly():
    c = PsfCorrection("f", SUB, NO_TEXTURE, 0.5, fluence=1e4, width_slope=2e-5)
    assert np.isclose(c.at_fluence(2e4).width ** 2, 0.25 + 0.2)
    assert np.isclose(c.at_fluence(0).width ** 2, 0.25 - 0.2)
    assert c.at_fluence(1e4) == c
    fainter = PsfCorrection("f", SUB, NO_TEXTURE, 0.5, 1e4, -3e-5)
    assert fainter.at_fluence(2e4).width == 0.0  # 0.25 - 0.3 < 0: clipped


def mock_stars(base, truth, n, width_of_fluence=None):
    """Stamps of point sources through `truth.apply(base)` within half a pixel
    of the centre, with read and shot noise; the fluence of each is its flux,
    and `width_of_fluence` broadens it."""
    model = _StarModel([blank_stamp(truth.apply(base))] * n, SUB)
    model.flux0 = 10 ** RNG.uniform(4, 5, n)
    model.background0, model.noise = np.full(n, 5.0), np.ones(n)
    x = zeros(model) | {"star_position": RNG.uniform(-1, 1, (n, 2))}
    widths = None
    if width_of_fluence is not None:
        widths = width_of_fluence(model.flux0)[:, None, None]
    clean = np.asarray(model(x, None, widths))
    std = np.sqrt(4.0 + clean.clip(0) / 2)
    data = clean + std * RNG.normal(size=clean.shape)
    mask = np.ones((SIZE, SIZE), bool)
    return [StarStamp(d, s, mask, base, f) for d, s, f in zip(data, std, model.flux0)]


def test_star_model_shifts_by_whole_pixels():
    model = _StarModel([blank_stamp(kernel())] * 2, SUB)
    model.flux0 = np.full(2, 1e4)
    # (dy, dx) = (0, 0) and (1, -2) detector pixels, latent = offset / 0.5
    shift = np.array([[0, 0], [2.0, -4.0]])
    stamps = np.asarray(model(zeros(model) | {"star_position": shift}))
    centred, moved = stamps - model.background0[:, None, None]
    inner = (slice(3, -3),) * 2
    np.testing.assert_allclose(
        moved[inner],
        np.roll(centred, (1, -2), (0, 1))[inner],
        atol=1e-10 * centred.max(),
    )


@pytest.mark.parametrize("k", [40, 41])
def test_star_model_crops_a_kernel_larger_than_the_canvas(k):
    small, big = kernel(k), np.pad(kernel(k), 60)  # canvas 3 * (11 + 24) = 105
    models = [_StarModel([blank_stamp(kk)], SUB) for kk in (small, big)]
    for m in models:
        m.flux0 = np.array([1e4])
    x = zeros(models[0]) | {"star_position": np.full((1, 2), 0.3)}
    a, b = (np.asarray(m(x, texture(9), 0.8)) for m in models)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-7 * a.max())


@pytest.mark.parametrize("k", [40, 41])
def test_learned_parts_match_the_applied_correction(k):
    """What the star fit learns is what `apply` gives the lens fit."""
    base = kernel(k)
    c = PsfCorrection("f", SUB, texture(9), 0.8)
    learning = _StarModel([blank_stamp(base)], SUB)
    frozen = _StarModel([blank_stamp(c.apply(base))], SUB)
    learning.flux0 = frozen.flux0 = np.array([1e4])
    x = {key: np.full(v.shape, 0.3) for key, v in learning.domain.items()}
    star = np.asarray(frozen(x)) - frozen(x | {"star_flux": np.full(1, -np.inf)})
    assert star.sum() > 1e3
    np.testing.assert_allclose(
        learning(x, c.log_texture, c.width), frozen(x), rtol=0, atol=1e-6 * star.max()
    )


def test_fit_recovers_an_injected_correction():
    base, radius = kernel(), 9
    truth = PsfCorrection("f", SUB, texture(radius), 0.8)
    stamps = mock_stars(base, truth, 24)
    learned, models = fit_psf_correction(
        stamps, SUB, "F", radius=radius, width=(0.5, 0.5), n_iterations=5
    )
    assert learned.filter == "f" and learned.log_texture.shape == (19, 19)
    error = np.abs(learned.apply(base) - truth.apply(base)).sum()
    untouched = np.abs(_center_even_axes(base) - truth.apply(base)).sum()
    assert error < untouched / 5
    residual = np.array([(s.data - m) / s.std for s, m in zip(stamps, models)])
    assert residual.std() < 1.1


def test_fit_recovers_a_width_growing_with_fluence():
    w0, slope = 0.4, 1e-5  # kernel pixels, pixels² per unit fluence
    identity = PsfCorrection("f", SUB, NO_TEXTURE, 0.0)
    stamps = mock_stars(kernel(), identity, 30, lambda f: np.sqrt(w0**2 + slope * f))
    learned, _ = fit_psf_correction(
        stamps, SUB, "f", width=(0.5, 0.5), fluence=5e4, n_iterations=5
    )
    assert learned.fluence == 5e4
    assert np.isclose(learned.width, np.sqrt(w0**2 + slope * 5e4), rtol=0.05)
    assert np.isclose(learned.at_fluence(0).width, w0, rtol=0.1)


def test_load_psf_kernel_applies_the_filter_correction(patched_seams, tmp_path):
    t = np.zeros((3, 3))
    t[1, 1] = 1.0
    # upper case, as a hand-built file may be; the kernel's filter is f444w
    PsfCorrection("F444W", 2, t, 0.0).save(tmp_path / "c.npz")
    raw = {"webbpsf_path": "", "psf_library": str(tmp_path)}
    raw["psf_arcsec_extension"] = 0.4
    data, centre = FakeJwstData("x.fits"), FakeJwstData("x.fits").wcs.center
    plain = jwst_kernel.load_psf_kernel(
        data, 2, centre, JwstPsfKernelConfig.from_yaml_dict(raw)
    )
    np.testing.assert_array_equal(plain, fake_build_webb_psf(None, None, None, None, 2))
    with_file = raw | {"correction": {"F444W": str(tmp_path / "c.npz")}}
    config = JwstPsfKernelConfig.from_yaml_dict(with_file)
    corrected = jwst_kernel.load_psf_kernel(data, 2, centre, config)
    expected = PsfCorrection.load(tmp_path / "c.npz").apply(plain)
    np.testing.assert_array_equal(corrected, expected)
    with pytest.raises(ValueError, match="for f444w at subsample 3"):
        jwst_kernel.load_psf_kernel(data, 3, centre, config)


def test_correction_of_an_unknown_filter_raises(tmp_path):
    cfg = make_config(tmp_path)
    cfg["telescope"]["psf"]["correction"] = {"f227w": "c.npz"}
    with pytest.raises(ValueError, match=r"\['f227w'\] match no filter"):
        ConfigParserJwst.from_yaml_dict(cfg)


def test_correction_needs_the_gaia_subsample_of_the_target(tmp_path):
    cfg = make_config(tmp_path, gaia=True, gaia_subsample=3)  # target subsample 2
    cfg["telescope"]["psf"]["correction"] = {"f444w": "c.npz"}
    with pytest.raises(ValueError, match=r"gaia_alignment.subsample \(3\)"):
        ConfigParserJwst.from_yaml_dict(cfg)


def test_fluence_needs_stars_of_different_fluence():
    stamps = [blank_stamp(kernel())] * 3  # every fluence 0
    with pytest.raises(ValueError, match="different StarStamp.fluence"):
        fit_psf_correction(stamps, SUB, "f", width=(0.5, 0.5), fluence=1e4)
    with pytest.raises(ValueError, match="fluence needs a width prior"):
        fit_psf_correction(stamps, SUB, "f", fluence=1e4)


def test_a_masked_border_keeps_the_starting_values_finite():
    stamp = mock_stars(kernel(), PsfCorrection("f", SUB, NO_TEXTURE, 0.0), 1)[0]
    y, x = np.indices((SIZE, SIZE)) - SIZE // 2
    round_mask = np.hypot(x, y) < SIZE // 2  # no border pixel left
    masked = StarStamp(stamp.data, stamp.std, round_mask, stamp.kernel)
    model = _StarModel([masked], SUB)
    assert np.isfinite(model.background0).all() and np.isfinite(model.flux0).all()


def test_a_stamp_without_unmasked_pixels_raises():
    good, empty = blank_stamp(kernel()), blank_stamp(kernel())
    empty = StarStamp(empty.data, empty.std, np.zeros((SIZE, SIZE), bool), empty.kernel)
    with pytest.raises(ValueError, match=r"stamps \[1\] have no unmasked pixel"):
        _StarModel([good, empty], SUB)
