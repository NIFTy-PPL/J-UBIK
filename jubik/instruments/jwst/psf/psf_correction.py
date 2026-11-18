# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

"""A learned correction of a filter's STPSF kernels, fitted on the field's stars.

    K' = G_w * N[K exp(t)]

``K`` is the STPSF kernel at a detector position (e.g. with ``opd_by_date``),
``t`` a log-texture on the kernel grid (a correlated field on a centred patch,
tapered to zero at its edge), ``N`` rescales to the sum of ``K`` and ``G_w`` is a
Gaussian of width ``w`` kernel pixels. One correction per filter, the same at
every detector position and in every exposure of the filter: it carries what
STPSF misses everywhere (wavefront residuals, charge diffusion, jitter), while
STPSF keeps the field dependence.

`fit_psf_correction` learns it from star stamps; the lens fit applies the stored
correction to its kernels (``psf.correction``, see `JwstPsfKernelConfig`), so
the arc cannot pull it: learning the kernel on the arc is degenerate with the
source. The lens fit keeps `PsfStatic`'s cached convolution.

Workflow, per filter:

1. Cut stamps of isolated, unsaturated stars from the filter's exposures, each
   with the STPSF kernel at its detector position (`load_psf_kernel` with the
   lens fit's PSF settings, no correction) and its peak per integration.
2. ``correction, models = fit_psf_correction(stamps, subsample, filter,
   radius=..., width=(mean, std), fluence=<the arc's peak per integration>)``;
   check the residuals of held-out stars (stars fitted with
   ``correction.at_fluence(peak).apply(kernel)`` and nothing learned).
3. ``correction.save(path)`` and ``psf: {correction: {<filter>: path}}``.

The texture's spectrum is fixed (see `log_texture_model`); choose its std and
slope on held-out stars. The width's variance may grow linearly with the stars'
peak (brighter-fatter); the stored correction is for one fluence.

Limits: the lens fit has one kernel per exposure, so every part of the image
(faint arc, lens light) gets the width of the stored fluence; brighter-fatter is
a per-pixel effect a convolution cannot carry. The texture may move the kernel's
centroid by a fraction of a kernel pixel, which the stars' free positions cannot
pin (B1938+666: up to 1 mas); the lens fit's shift must be free to take it up.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np

from .psf_operator import _center_even_axes


def corrected_kernel(kernel, log_texture):
    """``K exp(t)`` rescaled to the sum of ``K``, per kernel.

    ``kernel`` is (..., K, K), odd and centred on its middle pixel; ``log_texture``
    is (P, P), odd, P <= K, and is placed on the kernel's middle pixel (zero
    outside it).
    """
    if any(p > k for p, k in zip(log_texture.shape, kernel.shape[-2:])):
        raise ValueError(
            f"the texture, {log_texture.shape}, is larger than the kernel, "
            f"{kernel.shape[-2:]}"
        )
    shapes = zip(kernel.shape[-2:], log_texture.shape)
    pad = [((k - p) // 2,) * 2 for k, p in shapes]
    modified = kernel * jnp.exp(jnp.pad(log_texture, pad))
    total = (-2, -1)
    return modified * (
        kernel.sum(total, keepdims=True) / modified.sum(total, keepdims=True)
    )


def gaussian_transfer(shape: tuple[int, int], width):
    """The transfer function of a Gaussian of std `width` pixels on an FFT grid."""
    fy, fx = (np.fft.fftfreq(n) for n in shape)
    frequency2 = fy[:, None] ** 2 + fx[None, :] ** 2
    return jnp.exp(-2 * jnp.pi**2 * width**2 * frequency2)


@dataclass(frozen=True)
class PsfCorrection:
    """The learned correction of one filter's kernels, ``G_w * N[K exp(t)]``.

    Attributes
    ----------
    filter : str
        The filter, lower case.
    subsample : int
        Kernel pixels per detector pixel of the kernels it was learned on.
    log_texture : np.ndarray
        ``t``, (P, P) with P odd, on the kernel's middle pixel.
    width : float
        ``w``, the Gaussian's std in kernel pixels at `fluence`; 0 broadens
        nothing.
    fluence : float
        The peak per integration the width is for (0: one width for all).
    width_slope : float
        d(w²)/d(fluence), kernel pixels² per unit fluence (brighter-fatter).
    """

    filter: str
    subsample: int
    log_texture: np.ndarray
    width: float
    fluence: float = 0.0
    width_slope: float = 0.0

    def at_fluence(self, fluence: float) -> "PsfCorrection":
        """The correction for a source of peak `fluence` per integration."""
        variance = self.width**2 + self.width_slope * (fluence - self.fluence)
        return replace(
            self, width=float(np.sqrt(max(variance, 0.0))), fluence=float(fluence)
        )

    def apply(self, kernel: np.ndarray) -> np.ndarray:
        """The corrected kernel, odd and centred (even axes are recentred as in
        `PsfStatic`), with the sum of ``kernel``."""
        kernel = _center_even_axes(np.asarray(kernel, float))
        kernel = np.asarray(corrected_kernel(kernel, self.log_texture))
        if self.width > 0:
            transfer = np.asarray(gaussian_transfer(kernel.shape[-2:], self.width))
            kernel = np.fft.ifft2(np.fft.fft2(kernel) * transfer).real
        return kernel

    def save(self, path: str | Path) -> None:
        np.savez(path, **asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "PsfCorrection":
        with np.load(path) as f:
            return cls(**{k: f[k] if f[k].ndim else f[k].item() for k in f.files})


def log_texture_model(
    radius: int,
    fluctuations: float,
    loglogavgslope: float,
    prefix: str = "psf_texture",
) -> jft.Model:
    """``t``: a correlated field on a (2 radius + 1)² patch of kernel pixels,
    tapered by cos² from half the radius to zero at the radius.

    The taper hides the field's periodic wrap and keeps ``t`` where the stars
    constrain it. The spectrum is fixed: `fluctuations` is the field's std (and
    the zero mode's), `loglogavgslope` the power spectrum's log-log slope. A
    learned spectrum overfits in a MAP fit: at the stars' S/N the χ² gained by
    fitting noise outweighs any prior on the spectrum, and the texture turns
    rough at the pixel scale. Choose both on held-out stars.
    """
    size = 2 * radius + 1
    maker = jft.CorrelatedFieldMaker(prefix)
    # constants: their latents stay in the domain but do nothing
    maker.set_amplitude_total_offset(0.0, lambda _: fluctuations)
    maker.add_fluctuations(
        (size, size), 1.0, lambda _: fluctuations, lambda _: loglogavgslope
    )
    field = maker.finalize()
    r = np.hypot(*np.mgrid[-radius : radius + 1, -radius : radius + 1]) / radius
    taper = np.cos(np.pi * np.clip(r - 0.5, 0.0, 0.5)) ** 2
    return jft.Model(lambda x: taper * field(x), domain=field.domain, init=field.init)


@dataclass(frozen=True)
class StarStamp:
    """One star in one exposure.

    Attributes
    ----------
    data, std : np.ndarray
        (S, S) detector pixels, S odd, the star within half a pixel of the
        centre pixel; any unit, the same for both.
    mask : np.ndarray
        (S, S), True where a pixel is used.
    kernel : np.ndarray
        The base kernel at the star's detector position, ``subsample`` pixels
        per detector pixel, centred on (K - 1) / 2 as `PsfStatic` expects; the
        same shape for every stamp of a fit.
    fluence : float
        The star's peak per integration (e.g. DN), for a width that grows
        with brightness; see `fit_psf_correction`.
    """

    data: np.ndarray
    std: np.ndarray
    mask: np.ndarray
    kernel: np.ndarray
    fluence: float = 0.0


class _StarModel:
    """``flux bin(shift(G_w * N[K exp(t)])) + background + gradient`` per stamp.

    The kernel is shifted by a Fourier phase on a canvas `pad` detector pixels
    wider than the stamp on each side, then binned to detector pixels: an exact
    sub-pixel shift, so the correction does not learn an interpolation blur.
    """

    def __init__(self, stamps: list[StarStamp], subsample: int, pad: int = 12):
        self.data = np.stack([s.data for s in stamps]).astype(float)
        self.std = np.stack([s.std for s in stamps]).astype(float)
        self.mask = np.stack([s.mask for s in stamps]).astype(bool)
        if self.data.ndim != 3 or self.data.shape[1:] != self.std.shape[1:]:
            raise ValueError("the stamps must share one (S, S) shape")
        # an empty stamp's NaN start reaches the gradient and stalls the fit
        if empty := np.flatnonzero(~self.mask.any((1, 2))).tolist():
            raise ValueError(f"stamps {empty} have no unmasked pixel")
        size = self.data.shape[-1]
        if self.data.shape[-2] != size or size % 2 == 0:
            raise ValueError(f"stamps must be square and odd, got {self.data.shape}")
        self.kernels = np.stack(
            [_center_even_axes(np.asarray(s.kernel, float)) for s in stamps]
        )
        self.subsample, self.size, self.pad = subsample, size, pad
        self.fluence = np.array([s.fluence for s in stamps], float)

        n = subsample * (size + 2 * pad)
        k = self.kernels.shape[-1]
        lo = (k - 1) // 2 - n // 2  # the kernel's middle pixel lands on canvas n // 2
        self._crop = slice(max(lo, 0), min(lo + n, k))
        before = max(-lo, 0)
        self._pad = (before, n - (self._crop.stop - self._crop.start) - before)
        # canvas n // 2 to the middle of the stamp's centre pixel, in kernel pixels
        self._offset = subsample * (pad + size // 2) + (subsample - 1) / 2 - n // 2
        self._fy, self._fx = np.meshgrid(*(np.fft.fftfreq(n),) * 2, indexing="ij")
        self._shape = (n, n)
        self._yx = np.mgrid[:size, :size] - size // 2

        # starting values: background from the stamp's edge, flux from the rest
        edge = np.ones((size, size), bool)
        edge[1:-1, 1:-1] = False
        background = [
            np.median(d[edge & m] if (edge & m).any() else d[m])
            for d, m in zip(self.data, self.mask)
        ]
        self.background0 = np.array(background)
        signal = self.data - self.background0[:, None, None]
        flux = np.where(self.mask, signal, 0).sum((1, 2))
        self.flux0 = np.maximum(flux, 1e-3 * np.abs(flux).max())
        self.noise = np.nanmedian(np.where(self.mask, self.std, np.nan), axis=(1, 2))

    @property
    def domain(self):
        n = len(self.data)
        return {
            "star_flux": jft.ShapeWithDtype((n,)),
            "star_position": jft.ShapeWithDtype((n, 2)),
            "star_background": jft.ShapeWithDtype((n,)),
            "star_gradient": jft.ShapeWithDtype((n, 2)),
        }

    def _canvas(self, kernels):
        rows = kernels[..., self._crop, self._crop]
        return jnp.pad(rows, ((0, 0), self._pad, self._pad))

    def __call__(self, x, log_texture=None, width=None):
        """The (N, S, S) model stamps of the standard-normal star latents in `x`;
        `width` is shared or (N, 1, 1), one per stamp."""
        kernels = self.kernels
        if log_texture is not None:
            kernels = corrected_kernel(kernels, log_texture)
        transfer = 1.0 if width is None else gaussian_transfer(self._shape, width)
        canvas_fft = jnp.fft.fft2(self._canvas(kernels)) * transfer
        flux = self.flux0 * jnp.exp(x["star_flux"])
        dyx = 0.5 * x["star_position"]  # (dy, dx) in detector pixels
        background = self.background0 + 10 * self.noise * x["star_background"]
        gradient = self.noise[:, None] * x["star_gradient"]  # per pixel along (y, x)

        s, m, w = self.subsample, self.subsample * self.pad, self.subsample * self.size

        def stamp(k_fft, f, d, b, g):
            sy, sx = self._offset + s * d
            phase = jnp.exp(-2j * jnp.pi * (self._fy * sy + self._fx * sx))
            over = jnp.fft.ifft2(k_fft * phase).real[m : m + w, m : m + w]
            binned = over.reshape(self.size, s, self.size, s).sum((1, 3))
            return f * binned + b + g[0] * self._yx[0] + g[1] * self._yx[1]

        return jax.vmap(stamp)(canvas_fft, flux, dyx, background, gradient)


def fit_psf_correction(
    stamps: list[StarStamp],
    subsample: int,
    filter: str,
    *,
    radius: int | None = None,
    fluctuations: float = 0.1,
    loglogavgslope: float = -2.0,
    width: tuple[float, float] | None = None,
    fluence: float | None = None,
    n_iterations: int = 10,
) -> tuple[PsfCorrection, np.ndarray]:
    """Learn the correction of one filter's kernels on its stars (MAP).

    Every stamp is fitted with its kernel and its own flux, sub-pixel position,
    background and background gradient; the texture and the width are shared
    by all stamps. Stars of every exposure of the filter go into one fit.

    Bright stars are broader (brighter-fatter): with a `fluence`, the width's
    variance is linear in each stamp's ``fluence``, ``w² = w0² + (w1² - w0²)
    F / fluence``; the correction is for `fluence` (e.g. the arc's peak per
    integration), with width ``w1`` and the slope for `PsfCorrection.at_fluence`.
    The texture stays shared.

    Parameters
    ----------
    stamps : list[StarStamp]
        The stars, each with the base kernel at its position.
    subsample : int
        Kernel pixels per detector pixel.
    filter : str
        Stored in the correction.
    radius : int | None
        The texture's radius in kernel pixels (about the stamps' half width
        times `subsample`); None learns no texture.
    fluctuations, loglogavgslope : float
        The texture's fixed std and spectral slope, see `log_texture_model`;
        the defaults passed held-out stars of B1938+666 in F115W and F277W.
    width : tuple[float, float] | None
        Mean and std of the log-normal prior of the Gaussian's width in kernel
        pixels (of ``w0`` and ``w1``); None learns no broadening.
    fluence : float | None
        The fluence the correction is for, in the unit of ``StarStamp.fluence``;
        None shares one width among all stamps.
    n_iterations : int
        Newton-CG rounds of `jft.optimize_kl` without samples.

    Returns
    -------
    tuple[PsfCorrection, np.ndarray]
        The correction (identity parts where nothing is learned) and the best
        model of every stamp, (N, S, S). With neither `radius` nor `width`,
        only the stars are fitted: pass kernels corrected by
        `PsfCorrection.apply` to test a correction on other stars.
    """
    stars = _StarModel(stamps, subsample)
    texture = (
        None
        if radius is None
        else log_texture_model(radius, fluctuations, loglogavgslope)
    )
    sigma = widths = w0 = None
    if width is not None:
        sigma = widths = jft.LogNormalPrior(*width, name="psf_width", shape=())
    if width is not None and fluence is not None:
        if fluence <= 0:
            raise ValueError("fluence must be > 0; None shares one width")
        if np.ptp(stars.fluence) == 0:
            raise ValueError("fluence needs stamps of different StarStamp.fluence")
        w0, ratio = sigma, stars.fluence[:, None, None] / fluence
        sigma = jft.LogNormalPrior(*width, name="psf_width_at_fluence", shape=())

        def per_stamp(x):
            variance = w0(x) ** 2 + (sigma(x) ** 2 - w0(x) ** 2) * ratio
            return jnp.sqrt(jnp.maximum(variance, 1e-12))  # w1 < w0 may cross 0

        widths = jft.Model(per_stamp, domain=w0.domain | sigma.domain)

    domain = dict(stars.domain)
    for part in (texture, widths):
        if part is not None:
            domain |= part.domain

    def model(x):
        return stars(
            x,
            None if texture is None else texture(x),
            None if widths is None else widths(x),
        )

    models = jft.Model(model, domain=domain, white_init=True)
    mask, std = stars.mask, stars.std[stars.mask]
    likelihood = jft.Gaussian(
        stars.data[mask],
        noise_cov_inv=lambda r: r / std**2,
        noise_std_inv=lambda r: r / std,
    ).amend(jft.Model(lambda x: models(x)[mask], domain=domain, white_init=True))

    position = jft.Vector(jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), domain))
    samples, _ = jft.optimize_kl(
        likelihood,
        position,
        key=jax.random.PRNGKey(0),  # no samples: the key is unused
        n_total_iterations=n_iterations,
        n_samples=0,
        kl_kwargs=dict(
            minimize_kwargs=dict(name=None, maxiter=20, cg_kwargs=dict(name=None))
        ),
    )
    x = samples.pos.tree
    correction = PsfCorrection(
        filter=filter.lower(),
        subsample=int(subsample),
        log_texture=np.zeros((1, 1)) if texture is None else np.asarray(texture(x)),
        width=0.0 if sigma is None else float(sigma(x)),
    )
    if w0 is not None:
        slope = (correction.width**2 - float(w0(x)) ** 2) / fluence
        correction = replace(correction, fluence=float(fluence), width_slope=slope)
    return correction, np.asarray(models(x))
