import nifty8 as ift
import nifty8.re as jft

import jax
from jax import numpy as jnp

from astropy import units as u
from ..parse.sky_model.resolve_point_sources import ResolvePointSourcesModel


def jax_insert(x, ptsx, ptsy, bg):
    bg = bg.at[:, :, :, ptsx, ptsy].set(x)
    return bg


def resolve_point_sources(
    sky_dom: ift.RGSpace,
    resolve_point_souces_model: ResolvePointSourcesModel,
    bg=None
):
    if resolve_point_souces_model.freq_mode == "single":
        if resolve_point_souces_model.polarization == "I":
            if resolve_point_souces_model.mode == "fixed_locations":

                indsx, indsy = resolve_point_souces_model.locations.to_indices(
                    sky_dom, unit=u.rad)
                shp = sky_dom.shape
                full_shp = (1, 1, 1) + shp

                inv_gamma = jft.InvGammaPrior(
                    a=resolve_point_souces_model.a,
                    scale=resolve_point_souces_model.scale,
                    name="points",
                    shape=jax.ShapeDtypeStruct((len(indsx),), float),
                )
                if bg is None:
                    bg = jnp.zeros(full_shp)

                    def pts_func(x): return jax_insert(
                        inv_gamma(x), ptsx=indsx, ptsy=indsy, bg=bg
                    )
                    dom = inv_gamma.domain
                else:
                    def pts_func(x): return jax_insert(
                        inv_gamma(x), ptsx=indsx, ptsy=indsy, bg=bg(x)
                    )
                    dom = {**inv_gamma.domain, **bg.domain}
                pts_model = jft.Model(pts_func, domain=dom)
                additional = {}
                return pts_model, additional
    raise NotImplementedError("FIXME: Not Implemented for selected mode.")
