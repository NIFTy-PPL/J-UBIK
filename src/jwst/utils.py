import gwcs
from astropy.coordinates import SkyCoord

import nifty8.re as jft
import jax.numpy as jnp


def get_pixel(data_wcs: gwcs.wcs, location: SkyCoord, tol=1e-7) -> tuple:
    return data_wcs.numerical_inverse(location, with_units=True, tolerance=tol)


def build_sky_model(shape, dist, offset, fluctuations):
    cfm = jft.CorrelatedFieldMaker(prefix='reco')
    cfm.set_amplitude_total_offset(**offset)
    cfm.add_fluctuations(
        [int(shp) for shp in shape], dist,
        **fluctuations, non_parametric_kind='power')
    log_diffuse = cfm.finalize()

    def diffuse(x):
        return jnp.exp(log_diffuse(x))

    return jft.Model(diffuse, domain=log_diffuse.domain)


def build_shift_model(key, mean_sigma):
    from charm_lensing.models.parametric_models.parametric_prior import (
        build_prior_operator)
    distribution_model_key = ('normal', *mean_sigma)
    shape = (2,)

    shift_model = build_prior_operator(key, distribution_model_key, shape)
    domain = {key: jft.ShapeWithDtype((shape))}
    return jft.Model(shift_model, domain=domain)
