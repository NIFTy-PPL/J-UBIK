import gwcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as const

import nifty8.re as jft
import jax.numpy as jnp


def get_pixel(data_wcs: gwcs.wcs, location: SkyCoord, tol=1e-7) -> tuple:
    return data_wcs.numerical_inverse(location, with_units=True, tolerance=tol)


def build_sky_model(shape, dist, offset, fluctuations, extend_factor=1.5):
    assert len(shape) == 2

    cfm = jft.CorrelatedFieldMaker(prefix='reco')
    cfm.set_amplitude_total_offset(**offset)
    if 'non_parametric_kind' not in fluctuations:
        fluctuations['non_parametric_kind'] = 'power'
    cfm.add_fluctuations(
        [int(shp*extend_factor) for shp in shape], dist,
        **fluctuations)
    log_diffuse = cfm.finalize()

    # ext0, ext1 = [int(shp*extend_factor - shp)//2 for shp in shape]

    # def diffuse(x):
    #     return jnp.exp(log_diffuse(x)[ext0:-ext0, ext1:-ext1])

    ext0, ext1 = [int(shp*extend_factor - shp) for shp in shape]

    def diffuse(x):
        return jnp.exp(log_diffuse(x)[:-ext0, :-ext1])

    def full_diffuse(x):
        return jnp.exp(log_diffuse(x))

    return (jft.Model(diffuse, domain=log_diffuse.domain),
            jft.Model(full_diffuse, domain=log_diffuse.domain))
