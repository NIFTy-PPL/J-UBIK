# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""JAX spherical harmonic synthesis on HEALPix grids."""

import jax.numpy as jnp
import numpy as np
from nifty.re.correlated_field import get_sht


def get_healpix_sht(
    nside: int,
    lmax: int,
    mmax: int,
    *,
    backend: str = "ducc0",
    nthreads: int = 1,
):
    """Return a JAX callable for spin-0 HEALPix spherical harmonic synthesis.

    The callable maps a 1-D packed real harmonic vector to a real map of
    shape ``(12 * nside**2,)`` in HEALPix RING order. Band limits are inclusive;
    the input length is ``(lmax + 1)**2 - (lmax - mmax) * (lmax - mmax + 1)``.

    Coefficients follow NIFTy's convention: first the real m=0 coefficients
    for l=0..lmax, then interleaved real/imaginary pairs for l=m..lmax, ordered
    by increasing m. For m>0, each pair represents
    ``a_lm = (re + 1j * im) / sqrt(2)``. Synthesis includes NIFTy's ``sqrt(4*pi)``
    factor, so a unit monopole produces a map of ones.

    Both backends support ``jit``, ``jvp``, ``vjp`` and ``vmap``. Reverse-mode
    differentiation applies the Euclidean transpose of synthesis, without
    pixel-area weights; it is not an inverse spherical harmonic transform.

    Parameters
    ----------
    nside: int
        Positive HEALPix resolution parameter.
    lmax, mmax: int
        Harmonic band limits, with ``0 <= mmax <= lmax``.
    backend: str
        ``"ducc0"`` (default) runs on CPU and requires ``jubik[healpix]``.
        ``"cunusht"`` runs on CUDA, requires float64 input and
        ``mmax == lmax >= 32``. Its optional dependencies are installed with
        ``scripts/cunusht/install.sh``; see "HEALPix SHT on GPU" in the README.
    nthreads: int
        ducc0 threads; 0 uses all available hardware threads. Ignored by
        ``"cunusht"``.

    Returns
    -------
    callable
        Linear synthesis operator on packed real harmonic coefficients.
    """
    if nside < 1 or lmax < 0 or not 0 <= mmax <= lmax:
        raise ValueError("need nside >= 1 and 0 <= mmax <= lmax")
    if backend == "ducc0":
        return get_sht(nside, axis=0, lmax=lmax, mmax=mmax, nthreads=nthreads)
    if backend == "cunusht":
        return _get_cunusht_sht(nside, lmax, mmax)
    raise ValueError(f"backend must be 'ducc0' or 'cunusht', got {backend!r}")


def _require_float64(x):
    if x.dtype != jnp.float64:
        raise TypeError(
            f"the cunusht SHT backend needs float64 input, got {x.dtype}; enable "
            "it with jax.config.update('jax_enable_x64', True)"
        )


def _alm_from_real(x, alm, L):
    """Write nifty's packed real harmonic vector into complex, healpy-ordered
    `alm` (m >= 0). Works on numpy and cupy arrays."""
    alm[:L] = x[:L]
    alm[L:] = (x[L::2] + 1j * x[L + 1 :: 2]) / np.sqrt(2.0)


def _real_from_alm(alm, x, L):
    """Transpose of `_alm_from_real` under ducc0's alm inner product, which
    counts m > 0 twice: maps an adjoint synthesis back to the packed vector."""
    x[:L] = alm[:L].real
    x[L::2] = np.sqrt(2.0) * alm[L:].real
    x[L + 1 :: 2] = np.sqrt(2.0) * alm[L:].imag


_CUNUSHT_EPSILON = 1e-12


def _get_cunusht_sht(nside: int, lmax: int, mmax: int):
    """HEALPix synthesis on the GPU with cunuSHT's `nusht2d2` at the RING
    pixel centres, `nusht2d1` as its transpose, bound to JAX by jaxbind-gpu.
    Matches nifty's `get_sht` on 1-D packed real harmonic vectors."""
    import threading

    import jax
    from ducc0.healpix import Healpix_Base

    if mmax != lmax:
        raise ValueError(f"cunusht needs mmax == lmax, got {mmax=}, {lmax=}")
    if lmax < 32:
        # cunuSHT builds its internal grid from lmax; below lmax 32 it can have
        # fewer than 32 rings, and SHTns built with AVX-512 then calls exit(1).
        raise ValueError(f"cunusht needs lmax >= 32, got {lmax=}")
    # Before importing cupy and cuNuSHT: starting JAX's GPU backend loads its
    # pip CUDA libraries, which SHTns, cuFINUFFT and CuPy then reuse.
    try:
        jax.devices("gpu")
    except RuntimeError as e:
        raise RuntimeError(
            "the cunusht backend needs the CUDA build of jax and a visible GPU"
        ) from e
    x64 = jax.config.jax_enable_x64
    try:
        import cupy as cp
        from cunusht.deflection.GPU_nufft_transformer import (
            GPU_cufinufft_transformer,
        )
        from jaxbind_gpu import bind, current_context
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "the cunusht backend needs cupy, cufinufft, SHTns built with "
            "CUDA, the patched cunuSHT and jaxbind-gpu. Install them with "
            "scripts/cunusht/install.sh, see 'HEALPix SHT on GPU' in the README."
        ) from e
    finally:
        # cunuSHT switches x64 on at import; keep the caller's setting.
        jax.config.update("jax_enable_x64", x64)

    L = lmax + 1
    npix = 12 * nside**2
    n_real = L**2
    # (theta, phi) of the RING pixel centres, computed once.
    ptg = np.ascontiguousarray(Healpix_Base(nside, "RING").pix2ang(np.arange(npix)).T)
    # cunuSHT keeps a module-global timer and mutable work buffers.
    lock = threading.Lock()

    def transformer(nufft_type):
        # Build on the legacy default stream, where cufinufft's setpts runs:
        # uploaded on XLA's non-blocking stream, the points raced the type-1
        # bin sort (wrong adjoints and illegal addresses from nside 128 on).
        with cp.cuda.Stream.null:
            t = GPU_cufinufft_transformer(
                geominfo_deflection=("gl", {"lmax": lmax}),
                nuFFTtype=nufft_type,
                epsilon=_CUNUSHT_EPSILON,
            )
            # Setting `ptg` makes nusht2d{1,2} skip their own (inconsistent)
            # setpts. cufinufft's x runs along the first mode axis: phi for
            # type 2, theta for type 1 (as cunuSHT's lenmap2gclm does).
            t.ptg = cp.asarray(ptg)
            theta, phi = t.ptg
            t.nuFFTplan.setpts(*((phi, theta) if nufft_type == 2 else (theta, phi)))
            cp.cuda.Device().synchronize()
        return t

    def plan(nufft_type):
        # One plan per jaxbind callback, device and stream.
        key = ("cunusht", nside, lmax, nufft_type)
        return current_context().cached(key, lambda: transformer(nufft_type))

    # cunuSHT synchronizes the device around each of its steps, which orders
    # its own streams (SHTns, cufinufft, cupy) against XLA's.
    def forward(x, *, out):
        alm = cp.empty((n_real + L) // 2, dtype=cp.complex128)
        _alm_from_real(x, alm, L)
        with lock:
            t = plan(2)
            m = t.nusht2d2(lmax, lmax, alm=alm, loc=t.ptg.T, pointmap=alm[:0])
        cp.multiply(m, np.sqrt(4 * np.pi), out=out)
        current_context().retain(m)

    def adjoint(m, *, out):
        with lock:
            t = plan(1)
            alm = t.nusht2d1(
                lmax,
                lmax,
                pointmap=m.astype(cp.complex128),
                loc=t.ptg.T,
                epsilon=_CUNUSHT_EPSILON,
                alm=m[:0],
                verbose=0,
            )
        alm = np.sqrt(4 * np.pi) * alm.reshape(-1)
        _real_from_alm(alm, out, L)
        current_context().retain(alm)

    sht = bind(
        forward,
        output=lambda x: jax.ShapeDtypeStruct((npix,), x.dtype),
        out=True,
        transpose=adjoint,
    )

    def checked(x):
        _require_float64(x)
        if x.shape != (n_real,):
            raise ValueError(f"cunusht expects shape {(n_real,)}, got {x.shape}")
        return sht(x)

    return checked
