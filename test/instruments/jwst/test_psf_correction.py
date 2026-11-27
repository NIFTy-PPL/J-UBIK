"""The learned kernel correction `G_w * N[K exp(t)]` and its star fit."""

import jax
import nifty.re as jft
import numpy as np
import pytest
from scipy.special import ndtr

from jubik.instruments.jwst.parse.jwst_psf import JwstPsfKernelConfig
from jubik.instruments.jwst.parse.parsing_step import ConfigParserJwst
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.instruments.jwst.psf.psf_correction import (
    PsfCorrection,
    StarStamp,
    _StarModel,
    fit_psf_correction,
    log_texture_model,
    with_pixel_box,
)
from jubik.instruments.jwst.psf.psf_operator import PsfStatic, _center_even_axes
from jubik.instruments.jwst.rotation_and_shift.nufft_rotation_and_shift import (
    build_nufft_rotation_and_shift,
)

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
    c = PsfCorrection("f115w", SUB, RNG.normal(size=(7, 7)), 0.4, 2e4, 3e-6, True, 9e-7)
    c.save(tmp_path / "c.psf")  # the exact name, no .npz appended
    d = PsfCorrection.load(tmp_path / "c.psf")
    fields = ("filter", "subsample", "width", "fluence", "width_slope", "pixel_box")
    fields += ("brighter_fatter",)
    assert [getattr(d, f) for f in fields] == [getattr(c, f) for f in fields]
    np.testing.assert_array_equal(d.log_texture, c.log_texture)
    # a file written before pixel_box existed was learned without the box
    with open(tmp_path / "old.npz", "wb") as f:
        np.savez(f, filter="f", subsample=3, log_texture=NO_TEXTURE, width=0.4)
    old = PsfCorrection.load(tmp_path / "old.npz")
    assert old.pixel_box is False and old.brighter_fatter == 0.0
    # and a file without the box or kappa is one an older reader takes: no keys
    PsfCorrection("f", SUB, NO_TEXTURE, 0.4).save(tmp_path / "plain.npz")
    with np.load(tmp_path / "plain.npz") as f:
        assert "pixel_box" not in f.files and "brighter_fatter" not in f.files


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


@pytest.mark.parametrize("box", [False, True])
@pytest.mark.parametrize("k", [40, 41])
def test_learned_parts_match_the_applied_correction(k, box):
    """What the star fit learns is what `apply` (and the box, as
    `load_psf_kernel` adds it) gives the lens fit."""
    base = kernel(k)
    c = PsfCorrection("f", SUB, texture(9), 0.8)
    applied = with_pixel_box(c.apply(base)) if box else c.apply(base)
    learning = _StarModel([blank_stamp(base)], SUB, pixel_box=box)
    frozen = _StarModel([blank_stamp(applied)], SUB)
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
    nan = StarStamp(good.data.copy(), good.std, good.mask, good.kernel)
    nan.data[3, 4] = np.nan
    with pytest.raises(ValueError, match=r"stamps \[0\] have NaN data or std <= 0"):
        _StarModel([nan, good], SUB)


def gaussian_kernel(sigma, k):
    """A sampled Gaussian of std `sigma` kernel pixels, summing to one. Its
    transfer at the grid's Nyquist frequency is 3e-9 for sigma 2, 1.5e-5 for
    1.5; beyond 0.8 cycles per pixel, what a sky band of 0.2 meets, 5e-13."""
    y, x = np.mgrid[:k, :k] - (k - 1) / 2
    g = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return g / g.sum()


@pytest.mark.parametrize("box", [True, False])
def test_star_model_with_the_box_integrates_the_pixel(box):
    """A point source through a Gaussian PSF: with the box, the binned stamp is
    the PSF's integral over each detector pixel (erf), at any sub-pixel offset;
    without it, the sum of point samples is off by the box."""
    sigma, n = 2.0, 6  # kernel pixels; sigma / SUB detector pixels
    model = _StarModel(
        [blank_stamp(gaussian_kernel(sigma, 41))] * n, SUB, pixel_box=box
    )
    model.flux0, model.background0 = np.ones(n), np.zeros(n)
    latent = RNG.uniform(-1, 1, (n, 2))  # half a detector pixel at most
    stamps = np.asarray(model(zeros(model) | {"star_position": latent}))

    edges = np.arange(SIZE + 1) - SIZE // 2 - 0.5  # detector pixel edges
    s = sigma / SUB
    for stamp, (dy, dx) in zip(stamps, 0.5 * latent):
        py, px = (np.diff(ndtr((edges - d) / s)) for d in (dy, dx))
        error = np.abs(stamp - py[:, None] * px).max() / stamp.max()
        assert error < 1e-7 if box else error > 1e-3


@pytest.mark.parametrize("box", [True, False])
def test_pixel_box_makes_the_subsampled_response_the_pixel_integral(box):
    """A band-limited sky (sky-grid harmonics) through the subsampled response:
    NUFFT to the sub-pixel centres of a detector rotated by 20.8 deg, the
    boxed Gaussian kernel, the mean over each pixel's 3 x 3 sub-pixels. With
    the box this is the analytic mean of PSF * sky over the detector pixel,
    ``OTF(k) sinc(k_u) sinc(k_v)`` per harmonic along the detector axes."""
    n_sky, a, s, m = 64, 1.7, 3, 24  # sky px, sky px per detector px, subsample
    sigma = 1.5  # kernel (sub-)pixels: 0.5 detector px
    theta = np.radians(20.8)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # sky harmonics within 0.35 cycles per sky px, under the band limit of 0.4
    rng = np.random.default_rng(5)
    modes = []
    while len(modes) < 6:
        q = rng.integers(-22, 23, 2)
        if 0.05 < np.hypot(*q) / n_sky < 0.35:
            modes.append((q / n_sky, rng.uniform(0.5, 1), rng.uniform(0, 2 * np.pi)))
    yx = np.indices((n_sky, n_sky))

    def sky(points):
        return sum(
            c * np.cos(2 * np.pi * np.tensordot(k, points, 1) + p) for k, c, p in modes
        )

    # detector (u, v) to sky (y, x), centred on the sky
    origin = n_sky / 2 - a * rot @ np.full(2, (m - 1) / 2)
    u = (np.arange(m * s) + 0.5) / s - 0.5  # sub-pixel centres, detector px
    sub = origin[:, None, None] + a * np.tensordot(
        rot, np.stack(np.meshgrid(u, u, indexing="ij")), 1
    )
    nufft = build_nufft_rotation_and_shift(
        (n_sky, n_sky), (m * s,) * 2, band_limit=(0.4, 0.4)
    )
    fine = nufft(sky(yx), sub)
    k = gaussian_kernel(sigma, 25)  # +-8 sigma
    psf = PsfStatic(
        jft.ShapeWithDtype((m * s,) * 2, float), with_pixel_box(k) if box else k
    )
    model = np.asarray(psf((fine, {}))).reshape(m, s, m, s).mean((1, 3))

    centres = origin[:, None, None] + a * np.tensordot(rot, np.indices((m, m)), 1)
    exact = 0
    for kyx, c, p in modes:
        ku, kv = a * rot.T @ kyx  # cycles per detector px along the detector axes
        otf = np.exp(-2 * np.pi**2 * sigma**2 * (ku**2 + kv**2) / s**2)
        exact = exact + c * otf * np.sinc(ku) * np.sinc(kv) * np.cos(
            2 * np.pi * np.tensordot(kyx, centres, 1) + p
        )
    inner = (slice(5, -5),) * 2  # the kernel reaches 4 detector px beyond the cutout
    error = np.abs(model - exact)[inner].max()
    assert error < 1e-5 if box else error > 1e-3


def patch_weight(radius):
    k = kernel()
    c = (k.shape[0] - 1) // 2
    return k[c - radius : c + radius + 1, c - radius : c + radius + 1]


def moments(weight, t):
    """``sum K t b`` for b = 1, y, x, r², each over ``sum |K t b|``."""
    radius = (t.shape[0] - 1) // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return [
        float(abs((weight * t * b).sum()) / np.abs(weight * t * b).sum())
        for b in (np.ones_like(y), y, x, x**2 + y**2)
    ]


def test_orthogonal_texture_has_no_weighted_moments():
    radius = 9
    weight = patch_weight(radius)
    plain = log_texture_model(radius, 0.3, -2.0)
    orthogonal = log_texture_model(radius, 0.3, -2.0, weight=weight)
    assert plain.domain == orthogonal.domain
    for seed in range(3):
        x = plain.init(jax.random.PRNGKey(seed))
        t, t0 = np.asarray(orthogonal(x)), np.asarray(plain(x))
        assert max(moments(weight, t)) < 1e-10
        assert min(moments(weight, t0)) > 1e-3  # the plain field has all four
        # only a weighted fit by 1, y, x, r² is removed, before the taper
        r = np.hypot(*np.mgrid[-radius : radius + 1, -radius : radius + 1]) / radius
        taper = np.cos(np.pi * np.clip(r - 0.5, 0.0, 0.5)) ** 2
        inside = taper > 0.1
        y, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        basis = np.stack([np.ones_like(y), y, xx, xx**2 + y**2], -1)[inside]
        removed = ((t0 - t) / np.where(inside, taper, 1))[inside]
        _, rest, *_ = np.linalg.lstsq(basis, removed, rcond=None)
        assert np.sqrt(rest[0]) < 1e-10 * np.linalg.norm(removed)
        assert np.all(t[~(taper > 0)] == 0)  # still tapered to zero
    with pytest.raises(ValueError, match="is not the patch"):
        log_texture_model(radius, 0.3, -2.0, weight=patch_weight(radius - 1))


def test_texture_without_weight_is_the_tapered_field():
    radius = 9
    plain = log_texture_model(radius, 0.3, -2.0)
    maker = jft.CorrelatedFieldMaker("psf_texture")
    maker.set_amplitude_total_offset(0.0, lambda _: 0.3)
    maker.add_fluctuations((19, 19), 1.0, lambda _: 0.3, lambda _: -2.0)
    field = maker.finalize()
    r = np.hypot(*np.mgrid[-radius : radius + 1, -radius : radius + 1]) / radius
    taper = np.cos(np.pi * np.clip(r - 0.5, 0.0, 0.5)) ** 2
    x = plain.init(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(plain(x), taper * field(x))


def test_fit_with_the_box_learns_the_width_beyond_it():
    """Stars through the boxed kernel, fitted on the bare one: with the box the
    fit learns the true width; without it the width also takes the box's
    variance, 1/12 kernel px²."""
    base, w = kernel(), 0.3
    stamps = mock_stars(
        with_pixel_box(base), PsfCorrection("f", SUB, NO_TEXTURE, w), 24
    )
    stamps = [StarStamp(s.data, s.std, s.mask, base, s.fluence) for s in stamps]
    for box, expected in ((True, w), (False, np.sqrt(w**2 + 1 / 12))):
        learned, _ = fit_psf_correction(
            stamps, SUB, "f", width=(0.5, 0.5), n_iterations=3, pixel_box=box
        )
        assert np.isclose(learned.width, expected, rtol=0.05)


def test_fit_passes_the_options_on():
    """The fit's texture is orthogonal in the stamps' mean kernel (cut to the
    patch), and the correction records the box."""
    base, radius = kernel(), 9
    stamps = mock_stars(base, PsfCorrection("f", SUB, texture(radius), 0.8), 4)
    learned, _ = fit_psf_correction(
        stamps,
        SUB,
        "f",
        radius=radius,
        width=(0.5, 0.5),
        n_iterations=1,
        pixel_box=True,
        orthogonal=True,
    )
    assert learned.pixel_box is True
    assert np.abs(learned.log_texture).max() > 1e-4
    assert max(moments(patch_weight(radius), learned.log_texture)) < 1e-8
    with pytest.raises(ValueError, match="orthogonal needs a texture radius"):
        fit_psf_correction(stamps, SUB, "f", orthogonal=True)
    with pytest.raises(ValueError, match="larger than the kernel"):
        fit_psf_correction(stamps, SUB, "f", radius=21, orthogonal=True)  # kernel 41


def test_load_psf_kernel_adds_the_box_after_the_correction(patched_seams, tmp_path):
    raw = {"webbpsf_path": "", "psf_library": str(tmp_path)}
    raw["psf_arcsec_extension"] = 0.4
    data, centre = FakeJwstData("x.fits"), FakeJwstData("x.fits").wcs.center
    assert JwstPsfKernelConfig.from_yaml_dict(raw).pixel_box is False
    plain = jwst_kernel.load_psf_kernel(
        data, 2, centre, JwstPsfKernelConfig.from_yaml_dict(raw)
    )
    boxed = JwstPsfKernelConfig.from_yaml_dict(raw | {"pixel_box": True})
    np.testing.assert_array_equal(
        jwst_kernel.load_psf_kernel(data, 2, centre, boxed), with_pixel_box(plain)
    )
    np.testing.assert_allclose(with_pixel_box(plain).sum(), 1.0, rtol=1e-12)

    t = np.zeros((3, 3))
    t[1, 1] = 1.0
    for box in (True, False):
        PsfCorrection("f444w", 2, t, 0.3, pixel_box=box).save(tmp_path / f"{box}.npz")
    files = {True: str(tmp_path / "True.npz"), False: str(tmp_path / "False.npz")}
    config = JwstPsfKernelConfig.from_yaml_dict(
        raw | {"pixel_box": True, "correction": {"f444w": files[True]}}
    )
    expected = with_pixel_box(PsfCorrection.load(files[True]).apply(plain))
    np.testing.assert_array_equal(
        jwst_kernel.load_psf_kernel(data, 2, centre, config), expected
    )
    for box in (True, False):
        mismatch = raw | {"pixel_box": box, "correction": {"f444w": files[not box]}}
        with pytest.raises(ValueError, match="learned with pixel_box"):
            jwst_kernel.load_psf_kernel(
                data, 2, centre, JwstPsfKernelConfig.from_yaml_dict(mismatch)
            )
