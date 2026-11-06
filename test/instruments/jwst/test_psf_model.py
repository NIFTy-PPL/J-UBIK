"""PSF models with the `psf((image, params))` convention in the JWST response (#162)."""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
import scipy.optimize
from astropy import units as u
from jax.scipy.signal import fftconvolve

from jubik.instruments.jwst import jwst_response
from jubik.instruments.jwst.data.jwst_data import DataMetaInformation
from jubik.instruments.jwst.jwst_response import (
    TargetResponseInput,
    build_jwst_response,
    build_target_response,
)
from jubik.instruments.jwst.parse.jwst_response import SkyMetaInformation
from jubik.instruments.jwst.psf.psf_operator import (
    PsfGaussianBroadening,
    PsfKernelMix,
    PsfStatic,
    _center_even_axes,
    mixed_kernel,
)

N = 32
RNG = np.random.default_rng(162)


def width_model(shape=()):
    return jft.Model(
        lambda x: x["psf_width"], domain={"psf_width": jft.ShapeWithDtype(shape, float)}
    )


def gaussian_kernel(shape, sigma=1.5, offset=(0.3, -0.4)):
    y, x = (np.arange(n) - (n - 1) / 2 - o for n, o in zip(shape, offset))
    kernel = np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2) / sigma**2)
    return kernel / kernel.sum()


def centroid_from_middle(kernel):
    y, x = np.indices(kernel.shape)
    yx = np.array([(kernel * y).sum(), (kernel * x).sum()]) / kernel.sum()
    return yx - (np.array(kernel.shape) - 1) / 2


def response(psf, subsample=1, shape=(N, N)):
    sky = jft.Model(lambda x: x["sky"], domain={"sky": jft.ShapeWithDtype(shape)})
    return build_jwst_response(
        sky_in_subsampled_data=sky,
        data_meta=DataMetaInformation(
            unit=u.Jy, dvol=1.0 * u.arcsec**2, pixel_scale=None, color=None
        ),
        data_subsample=subsample,
        sky_meta=SkyMetaInformation(
            grid_extension=None, dvol=1.0 / subsample**2 * u.arcsec**2, unit=u.Jy
        ),
        psf=psf,
        zero_flux_model=None,
        data_mask=None,
    )


@pytest.mark.parametrize("shape", [(11, 11), (10, 10), (11, 10), (10, 11)])
@pytest.mark.parametrize("width", [0.0, 1e-4])
def test_zero_width_reproduces_static(shape, width):
    kernel = gaussian_kernel(shape)
    image = RNG.normal(size=(N, N))
    sky = jft.ShapeWithDtype((N, N), float)
    static = PsfStatic(sky, kernel)((image, {}))
    broadened = PsfGaussianBroadening(sky, kernel, width_model())(
        (image, {"psf_width": width})
    )
    np.testing.assert_allclose(broadened, static, atol=1e-8 * np.abs(static).max())


@pytest.mark.parametrize("shape", [(11, 11), (10, 10), (11, 10), (2, 10, 11)])
@pytest.mark.parametrize("width", [0.0, 0.3, 2.0, 50.0])
def test_sum_is_one_and_centroid_kept(shape, width):
    # A random kernel puts flux on the edges, where a circular convolution would wrap.
    kernel = RNG.uniform(size=shape)
    kernel /= kernel.sum(axis=(-2, -1), keepdims=True)
    sky = jft.ShapeWithDtype(shape[:-2] + (N, N))
    model = PsfGaussianBroadening(sky, kernel, width_model())
    broadened = np.asarray(model.broadened_kernel({"psf_width": width}))
    fixed = _center_even_axes(kernel)
    np.testing.assert_allclose(broadened.sum(axis=(-2, -1)), 1.0, rtol=1e-12)
    for b, f in zip(
        broadened.reshape(-1, *broadened.shape[-2:]),
        fixed.reshape(-1, *fixed.shape[-2:]),
    ):
        np.testing.assert_allclose(
            centroid_from_middle(b), centroid_from_middle(f), atol=1e-10
        )


def test_broadening_adds_the_gaussian_variance():
    kernel = gaussian_kernel((41, 41), sigma=1.5, offset=(0, 0))
    model = PsfGaussianBroadening(jft.ShapeWithDtype((N, N)), kernel, width_model())
    broadened = np.asarray(model.broadened_kernel({"psf_width": 2.0}))
    r = np.arange(broadened.shape[0]) - broadened.shape[0] // 2
    variance = (broadened.sum(axis=1) * r**2).sum()
    np.testing.assert_allclose(variance, 1.5**2 + 2.0**2, rtol=1e-6)


def test_response_domain_contains_psf_parameters():
    kernel = gaussian_kernel((11, 11))
    sky = jft.ShapeWithDtype((N, N))
    assert set(response(kernel).domain) == {"sky"}
    assert set(response(None).domain) == {"sky"}
    learned = response(PsfGaussianBroadening(sky, kernel, width_model()))
    assert set(learned.domain) == {"sky", "psf_width"}


def test_target_response_input_passes_psf_model(monkeypatch):
    built = {}
    monkeypatch.setattr(jwst_response, "build_sky_to_subsampled_data", lambda **_: None)
    monkeypatch.setattr(
        jwst_response, "build_jwst_response", lambda **kwargs: built.update(kwargs)
    )
    target_data = SimpleNamespace(
        psf=gaussian_kernel((11, 11)),
        mask=np.ones((1, 4, 4), bool),
        data=np.zeros((1, 4, 4)),
        subsample=1,
        subsample_centers=None,
    )
    config = dict(
        filter_name="f",
        grid=SimpleNamespace(spatial=None),
        sky_filter=SimpleNamespace(target={"f": None}),
        target_data=target_data,
        filter_meta=None,
        sky_meta=None,
        rotation_and_shift_algorithm=None,
        zero_flux_prior_configs=None,
        shift_and_rotation_correction=None,
    )

    build_target_response(TargetResponseInput(**config))
    np.testing.assert_array_equal(built["psf"], target_data.psf)

    psf_model = width_model()
    build_target_response(TargetResponseInput(**config, psf_model=psf_model))
    assert built["psf"] is psf_model


@pytest.mark.parametrize("n_obs", [None, 2])
def test_jit_and_grad(n_obs):
    shape = (11, 10) if n_obs is None else (n_obs, 11, 10)
    width_shape = () if n_obs is None else (n_obs,)
    sky_shape = (N, N) if n_obs is None else (n_obs, N, N)
    kernel = np.stack([gaussian_kernel(shape[-2:])] * (n_obs or 1)).reshape(shape)
    model = PsfGaussianBroadening(
        jft.ShapeWithDtype(sky_shape), kernel, width_model(width_shape)
    )
    learned = response(model, subsample=2, shape=sky_shape)
    x = {
        "sky": RNG.normal(size=sky_shape),
        "psf_width": np.linspace(0.7, 1.3, n_obs or 1).reshape(width_shape),
    }

    out = jax.jit(learned)(x)
    np.testing.assert_allclose(out, learned(x), rtol=1e-12)

    def loss(width):
        return (learned(x | {"psf_width": width}) ** 2).sum()

    grad = jax.grad(loss)(x["psf_width"])
    eps = 1e-6
    for i in np.ndindex(width_shape):
        step = np.zeros(width_shape)
        step[i] = eps
        fd = (loss(x["psf_width"] + step) - loss(x["psf_width"] - step)) / (2 * eps)
        np.testing.assert_allclose(grad[i], fd, rtol=1e-5)

    if n_obs is not None:  # each observation uses its own kernel and width
        for i in range(n_obs):
            single = PsfGaussianBroadening(
                jft.ShapeWithDtype((N, N)), kernel[i], width_model()
            )
            np.testing.assert_allclose(
                model((x["sky"], x))[i],
                single((x["sky"][i], {"psf_width": x["psf_width"][i]})),
                atol=1e-12,
            )


def test_fit_recovers_injected_width():
    rng = np.random.default_rng(0)
    true_width, subsample = 1.5, 2
    shape = (2 * N, 2 * N)
    sky = np.zeros(shape)
    rows, cols = rng.integers(8, 2 * N - 8, size=(2, 6))
    sky[rows, cols] = rng.uniform(50, 100, size=6)
    model = PsfGaussianBroadening(
        jft.ShapeWithDtype(shape), gaussian_kernel((15, 15), sigma=2.0), width_model()
    )
    learned = response(model, subsample=subsample, shape=shape)
    noise_std = 0.1  # peak ~10, statistical width error ~0.5%
    data = learned({"sky": sky, "psf_width": true_width})
    data = data + noise_std * rng.normal(size=data.shape)

    def chi2(width):
        residual = data - learned({"sky": sky, "psf_width": width[0]})
        return 0.5 * (residual**2).sum() / noise_std**2

    fit = scipy.optimize.minimize(
        jax.jit(jax.value_and_grad(chi2)), x0=[0.8], jac=True, method="L-BFGS-B"
    )
    assert fit.success
    np.testing.assert_allclose(fit.x[0], true_width, rtol=0.05)


def weight_model(shape=()):
    return jft.Model(lambda x: x["t"], domain={"t": jft.ShapeWithDtype(shape, float)})


def unit_kernels(shape, rng):
    """Two random kernels summing to one."""
    kernels = rng.uniform(size=(2,) + shape)
    return kernels / kernels.sum(axis=(-2, -1), keepdims=True)


# The centred kernels are 11 wide. On 20 px the grid is N + (K - 1) / 2 = 25 and 24 is
# 5-smooth too, so a grid one pixel short wraps into the crop; on 23 and 37 px the grid
# is padded (28 -> 30, 42 -> 45), so a crop placed on the padded grid fails.
@pytest.mark.parametrize("field_shape", [(20, 20), (23, 37)])
@pytest.mark.parametrize("shape", [(11, 11), (10, 10), (11, 10), (2, 10, 11)])
@pytest.mark.parametrize("t", [-0.5, 0.0, 0.3, 1.0, 1.5])
def test_mix_matches_fftconvolve_of_the_mixed_kernel(field_shape, shape, t):
    rng = np.random.default_rng(0)
    a, b = unit_kernels(shape, rng)
    field = rng.normal(size=shape[:-2] + field_shape)
    t = t + 0.25 * np.arange(2) if len(shape) == 3 else np.asarray(t)
    model = PsfKernelMix(jft.ShapeWithDtype(field.shape), a, b, weight_model(t.shape))

    centred_a, centred_b = _center_even_axes(a), _center_even_axes(b)
    same = partial(fftconvolve, mode="same")
    if len(shape) == 3:
        same = jax.vmap(same)

    def reference(f, t):
        t = t[..., None, None]
        return same(f, (1 - t) * centred_a + t * centred_b)

    cotangent = rng.normal(size=field.shape)

    def project(convolve):
        return lambda f, t: (convolve(f, t) * cotangent).sum()

    def mixed(f, t):
        return model((f, {"t": t}))

    np.testing.assert_allclose(mixed(field, t), reference(field, t), atol=1e-12)
    gradients = jax.grad(project(mixed), argnums=(0, 1))(field, t)
    expected = jax.grad(project(reference), argnums=(0, 1))(field, t)
    for g, e in zip(gradients, expected):
        np.testing.assert_allclose(g, e, atol=1e-12)


@pytest.mark.parametrize("shape", [(11, 11), (10, 10), (2, 11, 10)])
@pytest.mark.parametrize("t", [-0.5, 0.0, 0.3, 1.0, 1.5])
def test_mix_keeps_flux_and_centroid(shape, t):
    # point-symmetric, so centred on (K - 1) / 2
    kernels = unit_kernels(shape, np.random.default_rng(1))
    a, b = (kernels + kernels[..., ::-1, ::-1]) / 2
    point = np.zeros(shape[:-2] + (31, 31))
    point[..., 15, 15] = 1.0
    model = PsfKernelMix(jft.ShapeWithDtype(point.shape), a, b, weight_model())
    image = np.asarray(model((point, {"t": t})))
    np.testing.assert_allclose(image.sum(axis=(-2, -1)), 1.0, rtol=1e-12)
    for i in image.reshape(-1, 31, 31):
        np.testing.assert_allclose(centroid_from_middle(i), 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "second, match",
    [
        (np.full((2, 5, 6), 1 / 30), "differ in shape"),
        (np.full((2, 5, 5), 2 / 25), "sum to one"),
        (np.stack([np.full((5, 5), 1 / 25), np.full((5, 5), 2 / 25)]), "sum to one"),
    ],
)
@pytest.mark.parametrize("bad_first", [False, True])
def test_mix_refuses_kernels_of_another_shape_or_sum(second, match, bad_first):
    kernels = (np.full((2, 5, 5), 1 / 25), second)
    first, second = kernels[::-1] if bad_first else kernels
    with pytest.raises(ValueError, match=match):
        PsfKernelMix(jft.ShapeWithDtype((2, 8, 8)), first, second, weight_model())


def wrapped_gaussian(width, shape):
    """A sampled Gaussian wrapped onto a centred odd grid of `shape`, unit sum."""
    axes = []
    for size in shape:
        n = jnp.arange(size) - size // 2
        g = sum(jnp.exp(-0.5 * (n + m * size) ** 2 / width**2) for m in range(-4, 5))
        axes.append(g / g.sum())
    return axes[0][:, None] * axes[1][None, :]


@pytest.mark.parametrize("shape", [(11, 11), (10, 10), (2, 10, 11)])
@pytest.mark.parametrize("t", [-0.5, 0.3, 1.5])
def test_broadened_mix_matches_fftconvolve_of_the_built_kernel(shape, t):
    rng = np.random.default_rng(2)
    a, b = unit_kernels(shape, rng)
    field = rng.normal(size=shape[:-2] + (23, 37))
    t = t + 0.25 * np.arange(2) if len(shape) == 3 else np.asarray(t)
    sky = jft.ShapeWithDtype(field.shape)
    model = PsfGaussianBroadening(
        sky, mixed_kernel(a, b, weight_model(t.shape)), width_model()
    )
    centred_a, centred_b = _center_even_axes(a), _center_even_axes(b)
    same = partial(fftconvolve, mode="same")
    if len(shape) == 3:
        same = jax.vmap(same)

    def reference(f, t, w):
        t = t[..., None, None]
        mix = (1 - t) * centred_a + t * centred_b
        gauss = jnp.broadcast_to(wrapped_gaussian(w, mix.shape[-2:]), mix.shape)
        return same(f, fftconvolve(mix, gauss, mode="full", axes=(-2, -1)))

    def broadened(f, t, w):
        return model((f, {"t": t, "psf_width": w}))

    # width 0 is the plain mix; at 2.5 px the band-limited Gaussian of
    # PsfGaussianBroadening and the wrapped sampled one differ by ~ exp(-31)
    plain = PsfKernelMix(sky, a, b, weight_model(t.shape))((field, {"t": t}))
    np.testing.assert_allclose(broadened(field, t, 0.0), plain, atol=1e-12)
    w = jnp.asarray(2.5)
    np.testing.assert_allclose(
        broadened(field, t, w), reference(field, t, w), atol=1e-12
    )
    cotangent = rng.normal(size=field.shape)

    def project(convolve):
        return lambda f, t, w: (convolve(f, t, w) * cotangent).sum()

    gradients = jax.grad(project(broadened), argnums=(0, 1, 2))(field, t, w)
    expected = jax.grad(project(reference), argnums=(0, 1, 2))(field, t, w)
    for g, e in zip(gradients, expected):
        np.testing.assert_allclose(g, e, atol=1e-10)


def test_broadening_refuses_an_even_kernel_model():
    even = jft.Model(lambda _: gaussian_kernel((10, 10)), domain={})
    with pytest.raises(ValueError, match="odd and centred"):
        PsfGaussianBroadening(jft.ShapeWithDtype((N, N)), even, width_model())
