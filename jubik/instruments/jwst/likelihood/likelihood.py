from dataclasses import dataclass

import jax.numpy as jnp
import nifty.re as jft
import numpy as np

from ..variable_covariance.inverse_standard_deviation import InverseStdBuilder
from ....likelihood import build_gaussian_likelihood
from ..jwst_response import JwstResponse


# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


@dataclass
class GaussianLikelihoodBuilder:
    """Essential input data of `load_data`."""

    response: JwstResponse | jft.Model
    data: np.ndarray
    std: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)
        self.std = np.array(self.std)
        self.mask = np.array(self.mask)

    def build(self) -> jft.Gaussian:
        return build_gaussian_likelihood(
            jnp.array(np.array(self.data)[np.array(self.mask)], dtype=float),
            jnp.array(np.array(self.std)[np.array(self.mask)], dtype=float),
            model=self.response,
        )


@dataclass
class VariableCovarianceGaussianLikelihoodBuilder:
    """Essential input data of `load_data`."""

    response: JwstResponse | jft.Model
    inverse_std_builder: InverseStdBuilder
    data: np.ndarray
    std: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)
        self.std = np.array(self.std)
        self.mask = np.array(self.mask)

    def build(self) -> jft.VariableCovarianceGaussian:
        inv_std = self.inverse_std_builder.build(std=self.std, mask=self.mask)
        model = jft.Model(
            lambda x: (self.response(x), inv_std(x)),
            domain=self.response.domain | inv_std.domain,
        )
        like = jft.VariableCovarianceGaussian(self.data[self.mask])
        return like.amend(model, domain=jft.Vector(model.domain))


@dataclass
class ShotNoise:
    """Per-pixel σ² = a + b · max(model + sky, 0), in native data units.

    ``a`` is the variance at zero signal (read noise), ``b`` the shot-noise
    variance per unit of model, ``sky`` signal the reduction removed from the
    data but whose shot noise remains (0 if nothing was removed). Each is a
    scalar or has the data's shape.
    """

    a: np.ndarray | float
    b: np.ndarray | float
    sky: np.ndarray | float = 0.0


@dataclass
class ShotNoiseGaussianLikelihoodBuilder:
    """Gaussian with the mean and variance of Poisson plus read noise.

    The variance follows the response output R(s), σ² = a + b · max(R(s) +
    sky, 0), from the same response evaluation; no term depends on the data,
    so faint pixels are not down-weighted (Mighell 1999, ApJ 518, 380).
    """

    response: JwstResponse | jft.Model
    data: np.ndarray
    noise: ShotNoise
    mask: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)
        self.mask = np.array(self.mask)

    def model(self) -> jft.Model:
        """x -> (R(x), 1/σ(R(x))) on the unmasked pixels."""
        a, b, sky = (
            np.broadcast_to(np.asarray(v, float), self.mask.shape)[self.mask]
            for v in (self.noise.a, self.noise.b, self.noise.sky)
        )
        if not np.all(np.isfinite([a, b, sky]) & (a > 0) & (b >= 0)):
            raise ValueError("need finite a > 0, b >= 0 and sky on unmasked pixels")

        def response_and_inverse_std(x):
            r = self.response(x)
            return r, 1 / jnp.sqrt(a + b * jnp.maximum(r + sky, 0))

        return jft.Model(response_and_inverse_std, domain=self.response.domain)

    def build(self) -> jft.VariableCovarianceGaussian:
        model = self.model()
        like = jft.VariableCovarianceGaussian(
            jnp.array(self.data[self.mask], dtype=float)
        )
        return like.amend(model, domain=jft.Vector(model.domain))
