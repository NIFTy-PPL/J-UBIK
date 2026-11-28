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
source. The lens fit keeps `PsfStatic`'s cached convolution. Or the lens fit
learns it together with the stars (`joint_correction`, ``psf.joint_stars``):
the stars' likelihood joins the lens's and pins the kernel, with the full
star likelihood instead of an approximation of its posterior. With
``gaia_alignment`` the stars at Gaia positions sit there, moved by their
exposures' shifts, and pin the alignment too: one model per star.

Workflow, per filter:

1. Cut stamps of isolated, unsaturated stars from the filter's exposures, each
   with the STPSF kernel at its detector position (`load_psf_kernel` with the
   lens fit's PSF settings, no correction and no ``pixel_box``: the fit adds
   the box itself) and its peak per integration.
2. ``correction, models = fit_psf_correction(stamps, subsample, filter,
   radius=..., width=(mean, std), fluence=<the arc's peak per integration>)``;
   check the residuals of held-out stars (stars fitted with
   ``correction.at_fluence(peak).apply(kernel)``, nothing learned and
   ``pixel_box=correction.pixel_box``).
3. ``correction.save(path)`` and ``psf: {correction: {<filter>: path}}``.

The texture's spectrum is fixed (see `log_texture_model`); choose its std and
slope on held-out stars. The width's variance may grow linearly with the stars'
peak (brighter-fatter); the stored correction is for one fluence.

Three options of `fit_psf_correction`, off by default: ``pixel_box`` learns the
correction with the exact pixel integration (`with_pixel_box`), for a lens fit
with ``psf.pixel_box``; ``orthogonal`` keeps the texture out of what the
renormalisation, the stars' positions and the width already own (its weighted
constant, gradient and r² part), so the width alone carries the broadening;
``brighter_fatter`` moves the stars' charge between pixels (`charge_migration`),
for a lens fit with ``psf.brighter_fatter``: the width then keeps only the
broadening brighter-fatter does not explain, and its slope with fluence drops.

Limits: the lens fit has one kernel per exposure, so without
``brighter_fatter`` every part of the image (faint arc, lens light) gets the
width of the stored fluence. The texture may move the kernel's
centroid by a fraction of a kernel pixel, which the stars' free positions cannot
pin (a field's stars: up to 1 mas); the lens fit's shift must be free to take it up.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np

from ..integration.brighter_fatter import charge_migration
from .psf_operator import PsfKernelModel, _center_even_axes


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


def box_transfer(shape: tuple[int, int]) -> np.ndarray:
    """The transfer function of a box one pixel wide along each axis on an FFT
    grid, ``sinc(f_y) sinc(f_x)``."""
    fy, fx = (np.fft.fftfreq(n) for n in shape)
    return np.sinc(fy)[:, None] * np.sinc(fx)


def with_pixel_box(kernel: np.ndarray) -> np.ndarray:
    """`kernel` convolved with a box one kernel pixel wide along each axis, on
    its own grid (circular, as `PsfCorrection.apply`'s Gaussian).

    On a subsampled grid the box is one sub-pixel. For a band-limited kernel
    and sky, the sum of the sub-pixel samples of the (boxed kernel * sky) is
    then the exact integral over the detector pixel: each sub-pixel's integral
    is the box's mean at its centre. Without the box the sum is sharper by
    one sub-pixel box.
    """
    return np.fft.ifft2(np.fft.fft2(kernel) * box_transfer(kernel.shape[-2:])).real


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
    pixel_box : bool
        Learned with the sub-pixel box (`with_pixel_box`) in the star model,
        so its width does not stand in for it; the lens fit must then set
        ``psf.pixel_box``, see `JwstPsfKernelConfig`. Not part of `apply`.
    brighter_fatter : float
        ``kappa`` per DN per integration of the charge migration it was learned
        with (0: none); the lens fit must set the same
        ``psf.brighter_fatter``. Not part of `apply`.
    """

    filter: str
    subsample: int
    log_texture: np.ndarray
    width: float
    fluence: float = 0.0
    width_slope: float = 0.0
    pixel_box: bool = False
    brighter_fatter: float = 0.0

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
        fields = asdict(self)
        for key in ("pixel_box", "brighter_fatter"):
            if not fields[key]:  # unset, a file older jubik reads as before
                del fields[key]
        with open(path, "wb") as f:  # np.savez would append .npz to another suffix
            np.savez(f, **fields)

    @classmethod
    def load(cls, path: str | Path) -> "PsfCorrection":
        with np.load(path) as f:
            return cls(**{k: f[k] if f[k].ndim else f[k].item() for k in f.files})


def log_texture_model(
    radius: int,
    fluctuations: float,
    loglogavgslope: float,
    prefix: str = "psf_texture",
    weight: np.ndarray | None = None,
) -> jft.Model:
    """``t``: a correlated field on a (2 radius + 1)² patch of kernel pixels,
    tapered by cos² from half the radius to zero at the radius.

    The taper hides the field's periodic wrap and keeps ``t`` where the stars
    constrain it. The spectrum is fixed: `fluctuations` is the field's std (and
    the zero mode's), `loglogavgslope` the power spectrum's log-log slope. A
    learned spectrum overfits in a MAP fit: at the stars' S/N the χ² gained by
    fitting noise outweighs any prior on the spectrum, and the texture turns
    rough at the pixel scale. Choose both on held-out stars.

    With a `weight`, a kernel K on the patch ((2 radius + 1)², centred), ``t``
    is orthogonal to 1, y, x and r² in K's inner product, ``sum K t b = 0``:
    the field loses its weighted fit by those four before the taper. To first
    order ``K exp(t)`` then keeps K's sum, centroid and second moment, which
    the renormalisation, the stars' positions and the Gaussian's width own.
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
    if weight is None:
        return jft.Model(
            lambda x: taper * field(x), domain=field.domain, init=field.init
        )
    if np.shape(weight) != (size, size):
        raise ValueError(f"weight {np.shape(weight)} is not the patch, {(size,) * 2}")
    yx = np.mgrid[-radius : radius + 1, -radius : radius + 1] / radius
    basis = np.stack([np.ones_like(r), *yx, r**2]).reshape(4, -1).T
    weighted = basis * (weight * taper).reshape(-1, 1)  # t = taper f
    # f minus its weighted least-squares fit by the basis
    fit = jnp.asarray(np.linalg.solve(weighted.T @ basis, weighted.T))  # (4, size²)
    basis = jnp.asarray(basis)

    def orthogonal(x):
        f = field(x)
        return taper * (f - (basis @ (fit @ f.reshape(-1))).reshape(f.shape))

    return jft.Model(orthogonal, domain=field.domain, init=field.init)


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
    dn_per_unit : float | None
        DN per integration of one unit of ``data`` (EFFINTTM / PHOTMJSR for
        MJy/sr; 1 for DN per integration), which `fit_psf_correction`'s
        ``brighter_fatter`` needs.
    """

    data: np.ndarray
    std: np.ndarray
    mask: np.ndarray
    kernel: np.ndarray
    fluence: float = 0.0
    dn_per_unit: float | None = None


_STAMP_ARRAYS = ("data", "std", "mask", "kernel")


def save_star_stamps(
    path: str | Path, stamps: list[StarStamp], subsample: int, **extra
):
    """`stamps`, of one kernel shape, and the `subsample` of their kernels as one
    npz file for ``psf.joint_stars``; arrays in `extra`, one entry per stamp, are
    stored alongside. A lens fit with ``gaia_alignment`` needs ``exposure`` (the
    file name), ``x`` and ``y`` (the stamp's centre pixel in that file's data
    array) and ``ra`` and ``dec`` (deg) to tie stamps to Gaia stars, and also
    ``star`` (one id per star) to tie a star's stamps to the exposure shifts
    (``tied_position_std``)."""
    fields = {k: np.stack([getattr(s, k) for s in stamps]) for k in _STAMP_ARRAYS}
    fields["fluence"] = np.array([s.fluence for s in stamps], float)
    fields["dn_per_unit"] = np.array(
        [s.dn_per_unit for s in stamps], float
    )  # nan: None
    with open(path, "wb") as f:  # np.savez would append .npz to another suffix
        np.savez(f, subsample=subsample, **extra, **fields)


def load_star_stamps(path: str | Path) -> tuple[list[StarStamp], int, dict]:
    """The stamps, the kernels' subsample and the extra arrays of
    `save_star_stamps`."""
    own = (*_STAMP_ARRAYS, "fluence", "dn_per_unit")
    with np.load(path) as f:
        stamps = [StarStamp(*c) for c in zip(*(f[k] for k in own))]
        extra = {k: f[k] for k in f.files if k not in (*own, "subsample")}
        return stamps, int(f["subsample"]), extra


#: detector pixels of a star's canvas beyond its stamp on each side, see `_StarModel`
STAR_PAD = 12


class _StarModel:
    """``flux bin(shift(G_w * N[K exp(t)])) + background + gradient`` per stamp.

    The kernel is shifted by a Fourier phase on a canvas `pad` detector pixels
    wider than the stamp on each side, then binned to detector pixels: an exact
    sub-pixel shift, so the correction does not learn an interpolation blur.
    With `pixel_box` the kernel is convolved with one sub-pixel's box first
    (`with_pixel_box`, on the canvas), so the binning integrates the pixel
    exactly and the correction does not learn the box either. With
    `brighter_fatter` (kappa per DN per integration, or a model of it whose
    latents join the domain) each model stamp, in DN per integration by its
    ``dn_per_unit``, goes through `charge_migration`. `prefix` goes in front of
    the stars' latent names, e.g. one filter's in a lens fit. `positions`,
    ``(indices, model)``, places the stamps at `indices` by a model of their
    (n, 2) positions in subsampled pixels of each stamp, (x, y) with astropy's
    pixel centres (e.g. Gaia's, `gaia_star_location`), instead of by their own
    latents. `pad` must cover the stars' distance from the centre pixel.
    """

    def __init__(
        self,
        stamps: list[StarStamp],
        subsample: int,
        pad: int = STAR_PAD,
        pixel_box: bool = False,
        brighter_fatter: float | jft.Model | None = None,
        prefix: str = "",
        positions: tuple[np.ndarray, jft.Model] | None = None,
    ):
        self.data = np.stack([s.data for s in stamps]).astype(float)
        self.std = np.stack([s.std for s in stamps]).astype(float)
        self.mask = np.stack([s.mask for s in stamps]).astype(bool)
        if self.data.ndim != 3 or self.data.shape[1:] != self.std.shape[1:]:
            raise ValueError("the stamps must share one (S, S) shape")
        # an empty stamp's NaN start reaches the gradient and stalls the fit
        if empty := np.flatnonzero(~self.mask.any((1, 2))).tolist():
            raise ValueError(f"stamps {empty} have no unmasked pixel")
        bad = self.mask & ~(np.isfinite(self.data) & (self.std > 0))
        if bad := np.flatnonzero(bad.any((1, 2))).tolist():
            raise ValueError(
                f"stamps {bad} have NaN data or std <= 0 in unmasked pixels"
            )
        size = self.data.shape[-1]
        if self.data.shape[-2] != size or size % 2 == 0:
            raise ValueError(f"stamps must be square and odd, got {self.data.shape}")
        self.kernels = np.stack(
            [_center_even_axes(np.asarray(s.kernel, float)) for s in stamps]
        )
        self.subsample, self.size, self.pad = subsample, size, pad
        self.fluence = np.array([s.fluence for s in stamps], float)
        self.brighter_fatter, self.prefix = brighter_fatter, prefix
        self.positions = positions
        self._free = np.arange(len(stamps))
        if positions is not None:
            self._free = np.setdiff1d(self._free, positions[0])
        if brighter_fatter is not None:  # per unit of each stamp's data
            dn_per_unit = np.array([s.dn_per_unit for s in stamps], float)
            if not np.all(dn_per_unit > 0):  # None: nan
                raise ValueError(
                    "brighter_fatter needs every StarStamp.dn_per_unit > 0"
                )
            self._dn_per_unit = dn_per_unit[:, None, None]

        n = subsample * (size + 2 * pad)
        k = self.kernels.shape[-1]
        lo = (k - 1) // 2 - n // 2  # the kernel's middle pixel lands on canvas n // 2
        self._crop = slice(max(lo, 0), min(lo + n, k))
        before = max(-lo, 0)
        self._pad = (before, n - (self._crop.stop - self._crop.start) - before)
        # canvas n // 2 to the middle of the stamp's centre pixel, in kernel pixels
        self._offset = subsample * (pad + size // 2) + (subsample - 1) / 2 - n // 2
        # real canvases: the half spectrum along x (rfft)
        self._fy, self._fx = np.fft.fftfreq(n), np.fft.rfftfreq(n)
        self._shape = (n, n)
        self._box = box_transfer(self._shape)[:, : n // 2 + 1] if pixel_box else 1.0
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
        n, p = len(self.data), self.prefix
        domain = {
            f"{p}star_flux": jft.ShapeWithDtype((n,)),
            f"{p}star_background": jft.ShapeWithDtype((n,)),
            f"{p}star_gradient": jft.ShapeWithDtype((n, 2)),
        }
        if self._free.size:
            domain[f"{p}star_position"] = jft.ShapeWithDtype((self._free.size, 2))
        if self.positions is not None:
            domain |= self.positions[1].domain
        if isinstance(self.brighter_fatter, jft.Model):
            domain |= self.brighter_fatter.domain
        return domain

    def _dyx(self, x):
        """Each star's (dy, dx) from the middle of its stamp's centre pixel, in
        detector pixels."""
        free = 0.5 * x[f"{self.prefix}star_position"] if self._free.size else None
        if self.positions is None:
            return free
        indices, model = self.positions
        s = self.subsample
        dyx = ((model(x) + 0.5) / s - 0.5 - self.size // 2)[:, ::-1]
        out = jnp.zeros((len(self.data), 2), dyx.dtype).at[indices].set(dyx)
        return out if free is None else out.at[self._free].set(free)

    def _canvas(self, log_texture=None):
        """Each kernel's part on its canvas, `corrected_kernel` with a texture:
        exp(t) once, and each kernel's sums on the texture's patch alone."""
        rows = self.kernels[..., self._crop, self._crop]
        if log_texture is not None:
            k, p = self.kernels.shape[-1], log_texture.shape[-1]
            if p > k:
                raise ValueError(
                    f"the texture, {log_texture.shape}, is larger than the kernel, "
                    f"{self.kernels.shape[-2:]}"
                )
            lo = (k - p) // 2
            e = jnp.exp(log_texture)
            patch = self.kernels[..., lo : lo + p, lo : lo + p]
            total = self.kernels.sum((-2, -1), keepdims=True)
            moved = (patch * (e - 1)).sum((-2, -1), keepdims=True)
            texture = jnp.pad(e, (lo, k - p - lo), constant_values=1.0)
            rows = rows * texture[self._crop, self._crop] * (total / (total + moved))
        return jnp.pad(rows, ((0, 0), self._pad, self._pad))

    def __call__(self, x, log_texture=None, width=None):
        """The (N, S, S) model stamps of the standard-normal star latents in `x`;
        `width` is shared or (N, 1, 1), one per stamp."""
        canvas_fft = jnp.fft.rfft2(self._canvas(log_texture)) * self._box
        widths = jnp.zeros(len(self.data)) if width is None else width
        widths = jnp.broadcast_to(jnp.reshape(widths, (-1,)), (len(self.data),))
        p = self.prefix
        flux = self.flux0 * jnp.exp(x[f"{p}star_flux"])
        dyx = self._dyx(x)
        background = self.background0 + 10 * self.noise * x[f"{p}star_background"]
        gradient = self.noise[:, None] * x[f"{p}star_gradient"]  # per pixel, (y, x)

        s, m, w = self.subsample, self.subsample * self.pad, self.subsample * self.size

        def stamp(k_fft, f, d, b, g, sigma):
            # the Gaussian and the shift's phase, separable: G_w(y) G_w(x)
            sy, sx = self._offset + s * d
            row, col = (
                jnp.exp(-2 * jnp.pi * f_ * (jnp.pi * sigma**2 * f_ + 1j * shift))
                for f_, shift in ((self._fy, sy), (self._fx, sx))
            )
            over = jnp.fft.irfft2(k_fft * row[:, None] * col, s=self._shape)
            over = over[m : m + w, m : m + w]
            binned = over.reshape(self.size, s, self.size, s).sum((1, 3))
            return f * binned + b + g[0] * self._yx[0] + g[1] * self._yx[1]

        out = jax.vmap(stamp)(canvas_fft, flux, dyx, background, gradient, widths)
        kappa = self.brighter_fatter
        if isinstance(kappa, jft.Model):
            kappa = kappa(x)
        return (
            out if kappa is None else charge_migration(out, kappa * self._dn_per_unit)
        )


def _correction_model(
    stars: _StarModel,
    radius: int | None,
    fluctuations: float,
    loglogavgslope: float,
    width: tuple[float, float] | None,
    fluence: float | None,
    orthogonal: bool,
    prefix: str = "",
):
    """`fit_psf_correction`'s texture, the width at `fluence` (one for all
    without it), each stamp's width (N, 1, 1) and ``w0``, as models of latents
    named with `prefix`; None where nothing is learned."""
    texture = weight = None
    if orthogonal:
        if radius is None:
            raise ValueError("orthogonal needs a texture radius")
        c = (stars.kernels.shape[-1] - 1) // 2  # the kernels' middle pixel
        if radius > c:
            raise ValueError(
                f"the texture, {(2 * radius + 1,) * 2}, is larger than the kernel, "
                f"{stars.kernels.shape[-2:]}"
            )
        patch = slice(c - radius, c + radius + 1)
        weight = stars.kernels.mean(0)[patch, patch]
    if radius is not None:
        texture = log_texture_model(
            radius,
            fluctuations,
            loglogavgslope,
            prefix=f"{prefix}psf_texture",
            weight=weight,
        )
    if fluence is not None and width is None:
        raise ValueError("fluence needs a width prior")
    sigma = widths = w0 = None
    if width is not None:
        sigma = widths = jft.LogNormalPrior(*width, name=f"{prefix}psf_width", shape=())
    if width is not None and fluence is not None:
        if fluence <= 0:
            raise ValueError("fluence must be > 0; None shares one width")
        if np.ptp(stars.fluence) == 0:
            raise ValueError("fluence needs stamps of different StarStamp.fluence")
        w0 = sigma
        sigma = jft.LogNormalPrior(
            *width, name=f"{prefix}psf_width_at_fluence", shape=()
        )
        widths = _stamp_widths(stars, sigma, w0, fluence)
    return texture, sigma, widths, w0


def _stamp_widths(stars: _StarModel, sigma, w0, fluence):
    """Each stamp's width (N, 1, 1), ``w² = w0² + (w1² - w0²) F / fluence``, a
    model of the width ``w1`` at `fluence` (`sigma`) and `w0`; `sigma` itself
    without `w0`."""
    if w0 is None:
        return sigma
    ratio = stars.fluence[:, None, None] / fluence

    def per_stamp(x):
        variance = w0(x) ** 2 + (sigma(x) ** 2 - w0(x) ** 2) * ratio
        return jnp.sqrt(jnp.maximum(variance, 1e-12))  # w1 < w0 may cross 0

    return jft.Model(per_stamp, domain=w0.domain | sigma.domain)


def _star_likelihood(stars: _StarModel, texture, widths):
    """The Gaussian likelihood of the stamps' unmasked pixels and the model of
    every stamp, (N, S, S), with the `texture` and `widths` models (or None)."""
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
    return likelihood, models


def learned_kernel(kernels, texture, width, pixel_box: bool = False) -> jft.Model:
    """Each of `kernels` corrected, ``G_w * N[K exp(t)]``, then convolved with
    the box with `pixel_box`, as `PsfCorrection.apply` and `with_pixel_box` give
    it, as a model of the latents of `texture` and `width` (models of ``t`` and
    ``w``, None for an identity part)."""
    kernels = _center_even_axes(np.asarray(kernels, float))
    shape = kernels.shape[-2:]
    half = shape[-1] // 2 + 1  # rfft's columns; both transfers are even in f
    box = box_transfer(shape)[:, :half] if pixel_box else 1.0
    domain = {}
    for part in (texture, width):
        if part is not None:
            domain |= part.domain

    def kernel(x):
        k = kernels if texture is None else corrected_kernel(kernels, texture(x))
        if width is not None:
            transfer = box * gaussian_transfer(shape, width(x))[..., :half]
        else:
            transfer = box
        return jnp.fft.irfft2(jnp.fft.rfft2(k) * transfer, s=shape)

    return jft.Model(kernel, domain=domain)


@dataclass
class JointStarParts:
    """What a plot of `joint_correction` needs: each star set and the model of its
    stamps (N, S, S); the lens's base kernels (n, K, K) and its corrected kernels
    (a model); the texture, the width (at `fluence` with one), ``w0`` (the width
    at zero with a `fluence`) and kappa per DN per integration: models of the
    latents, a number or None."""

    stars: list
    base_kernels: np.ndarray
    kernel: jft.Model
    texture: jft.Model | None
    width: jft.Model | None
    w0: jft.Model | None
    fluence: float | None
    kappa: float | jft.Model | None


def joint_correction(
    stamps: list[StarStamp],
    subsample: int,
    kernels: np.ndarray,
    sky: jft.ShapeWithDtype,
    *,
    radius: int | None = None,
    fluctuations: float = 0.1,
    loglogavgslope: float = -2.0,
    width: tuple[float, float] | None = None,
    fluence: float | None = None,
    orthogonal: bool = False,
    pixel_box: bool = False,
    brighter_fatter: float | jft.Model | None = None,
    prefix: str = "",
    positions: tuple[np.ndarray, jft.Model] | None = None,
    more_stars: tuple[tuple[list[StarStamp], tuple, str, int], ...] = (),
) -> tuple[PsfKernelModel, list[jft.Likelihood], JointStarParts]:
    """`fit_psf_correction`'s correction, learned in a lens fit together with
    its stars instead of frozen into a file.

    Returns the convolution of the `sky` (n, H, W) by each of the base
    `kernels` (n, K, K) with the correction (`learned_kernel`; the width at
    `fluence` with one), and the stars' Gaussian likelihoods, the first of which
    `fit_psf_correction` maximizes. All are functions of the same texture,
    width and (if a model) kappa latents, named with `prefix`, so a sum of the
    lens likelihood and the stars' learns one correction from both. The options
    are `fit_psf_correction`'s; `brighter_fatter` may also be a model of kappa
    per DN per integration (learned): the stars' charge moves by it, and the
    lens's must move by the same model. `positions` places some of the
    `stamps` by a model (`_StarModel`), e.g. Gaia's. Each of `more_stars`,
    ``(stamps, positions, prefix, pad)``, is another set of stamps of one size
    under the same correction, with its own likelihood (e.g. Gaia stars without
    a PSF stamp). The third item holds the parts a plot needs (`JointStarParts`).
    """
    stars = _StarModel(
        stamps,
        subsample,
        pixel_box=pixel_box,
        brighter_fatter=brighter_fatter,
        prefix=prefix,
        positions=positions,
    )
    texture, sigma, widths, w0 = _correction_model(
        stars, radius, fluctuations, loglogavgslope, width, fluence, orthogonal, prefix
    )
    likelihood, models = _star_likelihood(stars, texture, widths)
    likelihoods, sets = [likelihood], [(stars, models)]
    for group, where, name, pad in more_stars:
        group = _StarModel(
            group,
            subsample,
            pad=pad,
            pixel_box=pixel_box,
            brighter_fatter=brighter_fatter,
            prefix=name,
            positions=where,
        )
        widths = _stamp_widths(group, sigma, w0, fluence)
        likelihood, models = _star_likelihood(group, texture, widths)
        likelihoods.append(likelihood)
        sets.append((group, models))
    kernel = learned_kernel(kernels, texture, sigma, pixel_box)
    parts = JointStarParts(
        sets, np.asarray(kernels), kernel, texture, sigma, w0, fluence, brighter_fatter
    )
    return PsfKernelModel(sky, kernel), likelihoods, parts


class _Converged(Exception):
    """`fit_psf_correction`'s stop at convergence, with the samples there."""

    def __init__(self, samples):
        super().__init__()
        self.samples = samples


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
    pixel_box: bool = False,
    orthogonal: bool = False,
    brighter_fatter: float | tuple[float, float] | None = None,
    n_samples: int = 0,
    seed: int = 0,
    converge: tuple[float, int] | None = None,
    history: list | None = None,
) -> (
    tuple[PsfCorrection, np.ndarray]
    | tuple[PsfCorrection, np.ndarray, list[PsfCorrection]]
):
    """Learn the correction of one filter's kernels on its stars (MAP, or
    variational with `n_samples`).

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
        the defaults passed a field's held-out stars in F115W and F277W.
    width : tuple[float, float] | None
        Mean and std of the log-normal prior of the Gaussian's width in kernel
        pixels (of ``w0`` and ``w1``); None learns no broadening.
    fluence : float | None
        The fluence the correction is for, in the unit of ``StarStamp.fluence``;
        None shares one width among all stamps.
    n_iterations : int
        Newton-CG rounds of `jft.optimize_kl`.
    pixel_box : bool
        Convolve every kernel with one sub-pixel's box (`with_pixel_box`), the
        exact pixel integration; the correction is then for a lens fit with
        ``psf.pixel_box``. Also for checking a correction on other stars.
    orthogonal : bool
        Make the texture orthogonal to 1, y, x and r² in the stamps' mean
        kernel (see `log_texture_model`'s weight), so it cannot take the
        kernel's sum, the stars' common shift or the width's broadening.
    brighter_fatter : float | tuple[float, float] | None
        ``kappa`` per DN per integration: every model stamp, in DN per
        integration by its `StarStamp.dn_per_unit`, goes through
        `charge_migration`, so the width keeps only the broadening
        brighter-fatter does not explain; the correction is then for a lens
        fit with ``psf.brighter_fatter`` at this kappa. Also for checking such
        a correction on other stars. A (mean, std) learns kappa with that
        log-normal prior (``psf_brighter_fatter``); the correction records
        its value. None moves no charge.
    n_samples : int
        Posterior samples of `jft.optimize_kl` (geoVI, each mirrored: 2
        `n_samples` in all); 0 fits the MAP.
    seed : int
        The fit's random key (the samples' draws).
    converge : tuple[float, int] | None
        ``(tol, patience)``: stop before `n_iterations` once the correction's
        latents (texture, widths, kappa) settle: an iteration's move is, per
        latent key, the rms over its elements of the change of the sample mean
        in sample stds (elements of zero std skipped), the largest over the
        keys; the fit stops after ``patience`` moves in a row below ``tol``.
        Needs `n_samples`; use 2 or more: one mirrored pair's stds are too
        noisy.
    history : list | None
        Each iteration appends a dict: ``iteration``, ``move`` (as for
        `converge`; inf at first, NaN for the MAP), and the
        posterior mean of ``texture_rms``, ``width`` (at the fluence),
        ``width_0`` and ``kappa``.

    Returns
    -------
    tuple[PsfCorrection, np.ndarray]
        The correction (identity parts where nothing is learned) at the latent
        mean, and the model of every stamp there, (N, S, S). With neither
        `radius` nor `width`, only the stars are fitted: pass kernels corrected
        by `PsfCorrection.apply` to test a correction on other stars. With
        `n_samples`, a third item: the correction at each of the 2 `n_samples`
        samples, whose spread (of ``width``, ``at_fluence(0).width``,
        ``width_slope``, the texture) is the correction's uncertainty.
    """
    kappa = brighter_fatter
    if isinstance(kappa, (tuple, list)):
        kappa = jft.LogNormalPrior(*kappa, name="psf_brighter_fatter", shape=())
    stars = _StarModel(stamps, subsample, pixel_box=pixel_box, brighter_fatter=kappa)
    texture, sigma, widths, w0 = _correction_model(
        stars, radius, fluctuations, loglogavgslope, width, fluence, orthogonal
    )
    likelihood, models = _star_likelihood(stars, texture, widths)
    domain = models.domain

    if converge is not None and not n_samples:
        raise ValueError("converge needs n_samples: the moves are in posterior stds")

    def correction_at(x):
        correction = PsfCorrection(
            filter=filter.lower(),
            subsample=int(subsample),
            log_texture=(
                np.zeros((1, 1)) if texture is None else np.asarray(texture(x))
            ),
            width=0.0 if sigma is None else float(sigma(x)),
            pixel_box=pixel_box,
            brighter_fatter=(
                0.0
                if kappa is None
                else float(kappa(x) if isinstance(kappa, jft.Model) else kappa)
            ),
        )
        if w0 is not None:
            slope = (correction.width**2 - float(w0(x)) ** 2) / fluence
            correction = replace(correction, fluence=float(fluence), width_slope=slope)
        return correction

    parts = [m for m in (texture, sigma, w0, kappa) if isinstance(m, jft.Model)]
    keys = sorted({k for m in parts for k in m.domain})
    last, calm = {}, [0]

    def callback(samples, state):
        trees = [t.tree for t in samples] if len(samples) else [samples.pos.tree]
        mean = {k: np.mean([t[k] for t in trees], 0) for k in keys}
        sd = {k: np.std([t[k] for t in trees], 0) for k in keys}
        per_key = []  # each key's rms of the change in sample stds, zero stds skipped
        for k in keys if last else ():
            change, std = np.ravel(mean[k] - last[k]), np.ravel(sd[k])
            if np.any(std > 0):
                ratio = change[std > 0] / std[std > 0]
                per_key.append(float(np.sqrt(np.mean(ratio**2))))
        move = max(per_key) if per_key else np.inf if len(samples) else np.nan
        last.update(mean)
        if history is not None:
            corrections = [correction_at(t) for t in trees]
            history.append(
                dict(
                    iteration=int(state.nit),
                    move=move,
                    texture_rms=float(
                        np.mean([c.log_texture.std() for c in corrections])
                    ),
                    width=float(np.mean([c.width for c in corrections])),
                    width_0=float(
                        np.mean([c.at_fluence(0).width for c in corrections])
                    ),
                    kappa=float(np.mean([c.brighter_fatter for c in corrections])),
                )
            )
        calm[0] = calm[0] + 1 if converge is not None and move < converge[0] else 0
        if converge is not None and calm[0] >= converge[1]:
            raise _Converged(samples)

    position = jft.Vector(jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), domain))
    quiet = dict(name=None, cg_kwargs=dict(name=None))
    try:
        samples, _ = jft.optimize_kl(
            likelihood,
            position,
            key=jax.random.PRNGKey(seed),
            n_total_iterations=n_iterations,
            n_samples=n_samples,
            draw_linear_kwargs=dict(cg_name=None, cg_kwargs={}),
            nonlinearly_update_kwargs=dict(minimize_kwargs=quiet),
            kl_kwargs=dict(minimize_kwargs=quiet | dict(maxiter=20)),
            callback=callback if converge is not None or history is not None else None,
        )
    except _Converged as stop:
        samples = stop.samples

    x = samples.pos.tree
    out = correction_at(x), np.asarray(models(x))
    return out + ([correction_at(s.tree) for s in samples],) if n_samples else out
