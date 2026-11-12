"""Residuals of every exposure of a band, back-projected onto the sky and stacked.

A single exposure is often noise-dominated: a misfit of a few percent of the
sky shows only once the residuals of all exposures are added up. For every
posterior sample x and likelihood, the masked residual weighted by the inverse
noise variance, ``(d - R_x(s_x)) / sigma^2``, goes back onto the band's sky
plane through the adjoint of the response at x, ``R_x^T``, with every nuisance
of the response (shift, PSF, zero flux, calibration) held at the sample's
values. For sigma held at the sample, that is the gradient of the
log-likelihood with respect to the band's sky: positive where the sky is too
faint. A band's stack is the sum over its
likelihoods, averaged over the samples.

Its noise level: pure-noise residuals, ``sigma N(0, 1) / sigma^2``, go
through the same adjoints, an equal share at each sample. ``std`` is their
pixel-wise standard deviation and ``stack / std`` the S/N map, Student-t with
one degree of freedom less than the draws: with the default of 100, ``|S/N| > 4``
is about twice as frequent as for a Gaussian, with 10 about 40 times.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np

from ..likelihood.likelihood import (
    GaussianLikelihoodBuilder,
    ShotNoiseGaussianLikelihoodBuilder,
)

if TYPE_CHECKING:
    from ..jwst_likelihoods import TargetLikelihoodProducts


@dataclass
class StackedResiduals:
    """A band's stacked residual and the pixel std of its pure-noise stacks."""

    stack: np.ndarray
    std: np.ndarray

    @property
    def snr(self) -> np.ndarray:
        """``stack / std``; nan where no unmasked pixel reaches the sky."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.std > 0, self.stack / self.std, np.nan)


def _inverse_std(builder):
    """x -> 1 / sigma on the unmasked pixels."""
    if isinstance(builder, ShotNoiseGaussianLikelihoodBuilder):
        model = builder.model()
        return lambda x: model(x)[1]
    if isinstance(builder, GaussianLikelihoodBuilder):
        inverse = jnp.asarray(1 / builder.std[builder.mask], dtype=float)
        return lambda x: inverse
    raise NotImplementedError(
        f"{type(builder).__name__}: stacked residuals need a noise without free parameters"
    )


def _back_projection(single):
    """(x, noise) -> R_x^T of the residual and of each noise row, one likelihood."""
    name, response, builder = single.filter, single.response, single.builder
    data = jnp.asarray(builder.data[builder.mask], dtype=float)
    inverse_std = _inverse_std(builder)

    @jax.jit
    def back_project(x, noise):
        prediction, adjoint = jax.vjp(lambda s: response(x | {name: s}), x[name])
        weight = inverse_std(x)
        rows = jnp.concatenate([((data - prediction) * weight**2)[None], noise * weight])
        return jax.vmap(lambda row: adjoint(row.astype(prediction.dtype))[0])(rows)

    return back_project, data.size


def stacked_residuals(
    products: "TargetLikelihoodProducts",
    sky: jft.Model,
    positions: Iterable[Mapping],
    bands: dict[str, list[str]] | None = None,
    n_noise: int = 100,
    seed: int = 0,
) -> dict[str, StackedResiduals]:
    """The stacked residual of every band, in the band's sky plane.

    Parameters
    ----------
    products : TargetLikelihoodProducts
        Its likelihoods' responses, data and noise, and the sky filter.
    sky : jft.Model
        Latent position -> the sky cube under the sky filter's key.
    positions : Iterable[Mapping]
        Posterior samples, also as ``jft.Vector`` (a ``jft.Samples`` works).
    bands : dict[str, list[str]] | None
        Band name -> the likelihoods (``filter`` of each) it sums; they must
        share a sky plane shape. None: one band per likelihood.
    n_noise : int
        Pure-noise stacks for ``std``, at least 2, spread over the samples:
        ``ceil(n_noise / n_samples)`` at each.
    seed : int
        Seed of the noise draws.

    Returns
    -------
    dict[str, StackedResiduals]
        Band name -> the stack, averaged over the samples, and its noise std.

    Raises
    ------
    NotImplementedError
        A likelihood whose noise has free parameters (variable covariance).
    """
    if n_noise < 2:
        raise ValueError(f"n_noise must be at least 2, got {n_noise}")
    positions = [getattr(x, "tree", x) for x in positions]
    if not positions:
        raise ValueError("no positions")
    singles = {single.filter: single for single in products.likelihoods}
    if bands is None:
        bands = {name: [name] for name in singles}
    back = {name: _back_projection(singles[name]) for names in bands.values() for name in names}
    per_sample = -(-n_noise // len(positions))
    rng = np.random.default_rng(seed)

    # shortcut: each noise draw goes through one sample's adjoint, not the sample average the stack takes, so std
    # is a little high (S/N low) where the samples' shifts or PSFs differ by a sky pixel or more; average every draw
    # over all samples, at n_samples times the cost, if that matters
    stack = {band: 0.0 for band in bands}
    # running sums of the noise stacks and their squares: the draws of all bands do not fit in memory on large skies
    s1, s2 = {band: 0.0 for band in bands}, {band: 0.0 for band in bands}
    for x in positions:
        x = dict(x) | products.sky_filter(sky(x))
        for band, names in bands.items():
            total = 0.0
            for name in names:
                back_project, size = back[name]
                total = total + np.asarray(back_project(x, rng.standard_normal((per_sample, size))))
            stack[band] = stack[band] + total[0] / len(positions)
            s1[band] = s1[band] + total[1:].sum(0)
            s2[band] = s2[band] + (total[1:] ** 2).sum(0)
    n = per_sample * len(positions)
    return {
        band: StackedResiduals(stack[band], np.sqrt(np.maximum(s2[band] - s1[band] ** 2 / n, 0) / (n - 1)))
        for band in bands
    }
