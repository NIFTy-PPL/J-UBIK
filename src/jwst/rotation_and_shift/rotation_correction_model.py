import nifty8.re as jft
from numpy.typing import ArrayLike
import jax.numpy as jnp

from ..parametric_model import build_parametric_prior


class RotationCorrection(jft.Model):
    def __init__(
        self,
        theta_prior: jft.Model,
        rotation_center: tuple[int, int],
    ):
        '''Rotation correction according to:
        ri = si Rot(theta) (pi - r)
            = Rot(theta) (si*pi -si*r),
        where si*r=rotation_center.
        '''
        self.rotation_center = rotation_center
        self.theta_prior = theta_prior

        # FIXME: HACK, the target shape is actually prescribed by the coords.
        # However, one does not necesserily know the coordinates shape upon
        # shift model creation.
        super().__init__(domain=theta_prior.domain, target=theta_prior.target)

    def __call__(self, params: dict, coords: ArrayLike) -> ArrayLike:
        theta = self.theta_prior(params)
        x = (jnp.cos(theta) * (coords[0]-self.rotation_center[0]) -
             jnp.sin(theta) * (coords[1]-self.rotation_center[1]))
        y = (jnp.sin(theta) * (coords[0]-self.rotation_center[0]) +
             jnp.cos(theta) * (coords[1]-self.rotation_center[1]))
        return jnp.array((x, y))


def build_rotation_correction_model(
    domain_key: str,
    mean_sigma: tuple[float, float],
    rotation_center: tuple[float, float],
) -> RotationCorrection:
    '''Rotation correction according to:
    ri = si Rot(theta) (pi - r)
        = Rot(theta) (si*pi -si*r),
    where si*r=rotation_center.

    Parameters
    ----------
    domain_key

    mean_sigma
        Mean and sigma of the Gaussian distribution for theta [rad]

    rotation_center
        The rotation center, has to be relative to the coords which will be
        supplied to the RotationCorrection.
    '''

    shape = (1,)

    rotation_prior = build_parametric_prior(
        domain_key, ('normal', *mean_sigma), shape)

    rotation_prior_model = jft.Model(
        rotation_prior, domain={domain_key: jft.ShapeWithDtype(shape)})

    return RotationCorrection(rotation_prior_model, rotation_center)
