"""GPU regressions for persistent plans; CPU-only installations skip this module."""

import gc
import importlib.util
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from jubik.instruments.resolve.cufinufft import PlanSet, nufft1, nufft2
from jubik.instruments.resolve.parse.response import CufinufftSettings
from jubik.instruments.resolve.response import (
    CufinufftResponse,
    interferometry_response_cufinufft,
    interferometry_response_finufft,
)


@pytest.fixture(scope="module", autouse=True)
def cuda_backend():
    try:
        devices = jax.devices("cuda")
    except RuntimeError:
        pytest.skip("CUDA-enabled JAX is required")
    for module in ("cufinufft", "jubik.instruments.resolve.cufinufft._exec"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed/built")
    with jax.enable_x64(True), jax.default_device(devices[0]):
        yield
    jax.clear_caches()
    gc.collect()


def fourier_matrix(shape, x, y):
    kx, ky = np.meshgrid(
        np.arange(shape[0]) - shape[0] // 2,
        np.arange(shape[1]) - shape[1] // 2,
        indexing="ij",
    )
    return np.exp(-1j * (x[:, None] * kx.ravel() + y[:, None] * ky.ravel()))


def assert_transform(actual, expected, tolerance):
    assert_allclose(
        actual, expected, rtol=tolerance,
        atol=tolerance * np.max(np.abs(expected)),
    )


@pytest.mark.parametrize("shape", [(12, 18), (13, 17)])
@pytest.mark.parametrize("dtype,eps,tolerance", [
    (np.complex128, 1e-10, 1e-8),
    (np.complex64, 1e-5, 1e-3),
])
def test_transforms_and_autodiff(shape, dtype, eps, tolerance):
    rng = np.random.default_rng(142)
    x, y = rng.uniform(-np.pi, np.pi, (2, 91))
    matrix = fourier_matrix(shape, x, y)
    plans = PlanSet(shape, x, y, eps=eps, dtype=dtype)
    sky = jnp.array((rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype))
    strengths = jnp.array((rng.normal(size=91) + 1j * rng.normal(size=91)).astype(dtype))

    def forward(sky):
        return nufft2(sky, plans)

    expected = matrix @ np.asarray(sky).ravel()
    assert_transform(forward(sky), expected, tolerance)
    assert_transform(jax.jit(forward)(sky), expected, tolerance)
    assert_transform(
        jax.jit(lambda c: nufft1(c, plans))(strengths),
        (matrix.conj().T @ strengths).reshape(shape), tolerance,
    )
    # JAX's complex transpose is bilinear: the VJP keeps the exponent sign.
    assert_transform(
        jax.vjp(forward, sky)[1](strengths)[0],
        (matrix.T @ strengths).reshape(shape), tolerance,
    )
    assert_transform(
        jax.jvp(forward, (sky,), (2 * sky,))[1], 2 * expected, tolerance,
    )
    assert_transform(
        jax.jit(jax.grad(lambda s: jnp.sum(jnp.abs(forward(s)) ** 2)))(sky),
        (2 * matrix.T @ expected.conj()).reshape(shape), tolerance,
    )

    skies = jnp.stack([sky, 2 * sky, 3 * sky])
    skies = jnp.stack([skies, 4 * skies])
    expected_batch = np.asarray(skies).reshape(2, 3, -1) @ matrix.T
    assert_transform(
        jax.jit(jax.vmap(jax.vmap(forward)))(skies), expected_batch, tolerance,
    )
    assert_transform(
        jax.jit(jax.vmap(forward, in_axes=1, out_axes=1))(skies[0].swapaxes(0, 1)),
        expected_batch[0].T, tolerance,
    )


def test_repeated_execution_reuses_plan():
    plans = PlanSet((12, 18), [.1, .2], [.3, .4], eps=1e-8)
    assert plans.n_plans == 0
    compiled = jax.jit(lambda sky: nufft2(sky, plans))
    sky = jnp.ones((12, 18), dtype=np.complex128)
    expected = fourier_matrix((12, 18), np.array([.1, .2]), np.array([.3, .4])) @ np.ones(216)
    for _ in range(10):
        assert_transform(compiled(sky), expected, 1e-6)
    assert plans.n_plans == 1


def test_executable_retains_and_releases_plans():
    def compile_response():
        plans = PlanSet((12, 18), [.1, .2], [.3, .4], eps=1e-8)
        compiled = jax.jit(lambda sky: nufft2(sky, plans)).lower(
            jax.ShapeDtypeStruct((12, 18), np.complex128),
        ).compile()
        return compiled, weakref.ref(plans)

    compiled, owner = compile_response()
    jax.clear_caches()
    gc.collect()
    # Check ownership before executing: the old implementation leaves dangling
    # pointers here and can crash the process if we call the executable.
    assert owner() is not None
    expected = fourier_matrix((12, 18), np.array([.1, .2]), np.array([.3, .4])) @ np.ones(216)
    assert_transform(compiled(np.ones((12, 18), np.complex128)), expected, 1e-6)
    del compiled
    jax.clear_caches()
    gc.collect()
    assert owner() is None


def test_close_releases_and_blocks_new_plans():
    plans = PlanSet((12, 18), [.1, .2], [.3, .4], eps=1e-8)
    expected = fourier_matrix((12, 18), np.array([.1, .2]), np.array([.3, .4])) @ np.ones(216)
    sky = jnp.ones((12, 18), dtype=np.complex128)
    assert_transform(jax.jit(lambda sky: nufft2(sky, plans))(sky), expected, 1e-6)
    assert plans.n_plans == 1
    plans.close()
    plans.close()
    assert plans.stream is None
    assert plans.n_plans == 0
    with pytest.raises(RuntimeError):
        plans.plan(2, -1, 1)


@pytest.mark.parametrize("center", [(0., 0.), (1e-4, -3e-4)])
def test_callable_response_matches_finufft(center):
    pytest.importorskip("jax_finufft")
    rng = np.random.default_rng(15)
    shape = (16, 24)
    observation = SimpleNamespace(
        freq=np.array([1e9, 1.3e9]),
        uvw=rng.uniform(-100, 100, (50, 3)),
    )
    geometry = dict(
        observation=observation,
        pixsize_x=2e-5,
        pixsize_y=3e-5,
        center_x=center[0],
        center_y=center[1],
    )
    response = interferometry_response_cufinufft(
        npix_x=shape[0], npix_y=shape[1],
        settings=CufinufftSettings(
            epsilon=1e-10, gpu_maxbatchsize=0, upsampfac=2.,
        ),
        **geometry,
    )
    reference = interferometry_response_finufft(epsilon=1e-10, **geometry)
    assert isinstance(response, CufinufftResponse)
    sky = jnp.array(rng.normal(size=shape))
    expected = jax.jit(reference)(sky)
    compiled = jax.jit(response).lower(sky).compile()
    assert expected.shape == (50, 2)
    assert_transform(response(sky), expected, 1e-8)
    assert_transform(compiled(sky), expected, 1e-8)
    skies = jnp.stack([sky, 2 * sky])
    assert_transform(
        jax.jit(jax.vmap(response))(skies),
        jax.jit(jax.vmap(reference))(skies), 1e-8,
    )

    def gradient(operator):
        return jax.jit(jax.grad(lambda s: jnp.sum(jnp.abs(operator(s)) ** 2)))(sky)

    assert_transform(gradient(response), gradient(reference), 1e-8)

    # Compiled code needs only the plans and captured constants, not the
    # response instance that supplied __call__ during tracing.
    owner = weakref.ref(response)
    plans = weakref.ref(response.plans)
    del response
    jax.clear_caches()
    gc.collect()
    assert owner() is None
    assert plans() is not None
    assert_transform(compiled(sky), expected, 1e-8)
    del compiled
    jax.clear_caches()
    gc.collect()
    assert plans() is None


@pytest.mark.parametrize("nufft_type", [1, 2])
@pytest.mark.parametrize("n_trans", [1, 3])
def test_concurrent_execution(nufft_type, n_trans):
    rng = np.random.default_rng(43)
    shape, n_points = (64, 96), 10000
    x, y = rng.uniform(-np.pi, np.pi, (2, n_points))
    # A batch larger than gpu_maxbatchsize also exercises repeated workspace
    # use inside a single execute call.
    plans = PlanSet(shape, x, y, eps=1e-9, gpu_maxbatchsize=1)
    transform = nufft1 if nufft_type == 1 else nufft2
    source_shape = (n_trans,) + ((n_points,) if nufft_type == 1 else shape)
    inputs = [jax.device_put(
        rng.normal(size=source_shape) + 1j * rng.normal(size=source_shape),
    ) for _ in range(4)]
    compiled = jax.jit(jax.vmap(lambda source: transform(source, plans))).lower(inputs[0]).compile()
    references = [np.asarray(compiled(source)) for source in inputs]
    barrier = Barrier(4, timeout=30)

    def worker(index):
        errors = []
        for _ in range(30):
            barrier.wait()
            output = np.asarray(compiled(inputs[index]))
            errors.append(np.linalg.norm(output - references[index]) / np.linalg.norm(references[index]))
        return max(errors)

    with ThreadPoolExecutor(4) as pool:
        errors = list(pool.map(worker, range(4)))
    assert max(errors) < 1e-8, errors
    assert plans.n_plans == 1
