# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Jakob Roth

import nifty8 as ift
import nifty8.re as jft
import jax
from jax import numpy as jnp
import numpy as np


# TODO: replace with new config parsing!
def _spatial_dom(cfg):
    nx = cfg.getint("space npix x")
    ny = cfg.getint("space npix y")
    dx = str2rad(cfg["space fov x"]) / nx
    dy = str2rad(cfg["space fov y"]) / ny
    return ift.RGSpace([nx, ny], [dx, dy])

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
SPEEDOFLIGHT = 299792458.
# TODO: replace with new astropy unit handling!
def str2rad(s):
    """Convert string of number and unit to radian.

    Support the following units: muas mas as amin deg rad.

    Parameters
    ----------
    s : str
        TODO

    """
    c = {
        "muas": AS2RAD * 1e-6,
        "mas": AS2RAD * 1e-3,
        "as": AS2RAD,
        "amin": ARCMIN2RAD,
        "deg": DEG2RAD,
        "rad": 1,
    }
    keys = list(c.keys())
    keys.sort(key=len)
    for kk in reversed(keys):
        nn = -len(kk)
        unit = s[nn:]
        if unit == kk:
            return float(s[:nn]) * c[kk]
    raise RuntimeError("Unit not understood")

def build_cf(prefix, conf, shape, dist):
    zmo = conf.getfloat(f"{prefix} zero mode offset")
    zmm = conf.getfloat(f"{prefix} zero mode mean")
    zms = conf.getfloat(f"{prefix} zero mode stddev")
    flum = conf.getfloat(f"{prefix} fluctuations mean")
    flus = conf.getfloat(f"{prefix} fluctuations stddev")
    llam = conf.getfloat(f"{prefix} loglogavgslope mean")
    llas = conf.getfloat(f"{prefix} loglogavgslope stddev")
    flem = conf.getfloat(f"{prefix} flexibility mean")
    fles = conf.getfloat(f"{prefix} flexibility stddev")
    aspm = conf.getfloat(f"{prefix} asperity mean")
    asps = conf.getfloat(f"{prefix} asperity stddev")

    cf_zm = {"offset_mean": zmo, "offset_std": (zmm, zms)}
    cf_fl = {
        "fluctuations": (flum, flus),
        "loglogavgslope": (llam, llas),
        "flexibility": (flem, fles),
        "asperity": (aspm, asps),
        "harmonic_type": "Fourier",
    }
    cfm = jft.CorrelatedFieldMaker(prefix)
    cfm.set_amplitude_total_offset(**cf_zm)
    cfm.add_fluctuations(
        shape, distances=dist, **cf_fl, prefix="", non_parametric_kind="power"
    )
    amps = cfm.get_normalized_amplitudes()
    cfm = cfm.finalize()
    additional = {f"ampliuted of {prefix}": amps}
    return cfm, additional


def sky_model_diffuse(cfg):
    if not cfg["freq mode"] == "single":
        raise NotImplementedError("FIXME: only implemented for single frequency")
    if not cfg["polarization"] == "I":
        raise NotImplementedError("FIXME: only implemented for stokes I")
    sky_dom = _spatial_dom(cfg)
    bg_shape = sky_dom.shape
    bg_distances = sky_dom.distances
    bg_log_diff, additional = build_cf(
        "stokesI diffuse space i0", cfg, bg_shape, bg_distances
    )
    full_shape = (1, 1, 1) + bg_shape

    def bg_diffuse(x):
        return jnp.broadcast_to(
            jnp.exp(bg_log_diff(x["stokesI diffuse space i0"])), full_shape
        )

    bg_diffuse_model = jft.Model(
        bg_diffuse, domain={"stokesI diffuse space i0": bg_log_diff.domain}
    )

    return bg_diffuse_model, additional


def jax_insert(x, ptsx, ptsy, bg):
    bg = bg.at[:, :, :, ptsx, ptsy].set(x)
    return bg


def sky_model_points(cfg, bg=None):
    if cfg["freq mode"] == "single":
        if cfg["polarization"] == "I":
            if cfg["point sources mode"] == "single":
                ppos = []
                sky_dom = _spatial_dom(cfg)
                s = cfg["point sources locations"]
                for xy in s.split(","):
                    x, y = xy.split("$")
                    ppos.append((str2rad(x), str2rad(y)))
                ppos = np.array(ppos)
                dx = np.array(sky_dom.distances)
                center = np.array(sky_dom.shape) // 2
                inds = np.unique(np.round(ppos / dx + center).astype(int).T, axis=1)
                indsx, indsy = inds
                alpha = cfg.getfloat("point sources alpha")
                q = cfg.getfloat("point sources q")
                sky_dom = _spatial_dom(cfg)
                shp = sky_dom.shape
                full_shp = (1, 1, 1) + shp
                inv_gamma = jft.InvGammaPrior(
                    a=alpha,
                    scale=q,
                    name="points",
                    shape=jax.ShapeDtypeStruct((len(indsx),), float),
                )
                if bg is None:
                    bg = jnp.zeros(full_shp)
                    pts_func = lambda x: jax_insert(
                        inv_gamma(x), ptsx=indsx, ptsy=indsy, bg=bg
                    )
                    dom = inv_gamma.domain
                else:
                    pts_func = lambda x: jax_insert(
                        inv_gamma(x), ptsx=indsx, ptsy=indsy, bg=bg(x)
                    )
                    dom = {**inv_gamma.domain, **bg.domain}
                pts_model = jft.Model(pts_func, domain=dom)
                additional = {}
                return pts_model, additional
    raise NotImplementedError("FIXME: Not Implemented for selected mode.")


def sky_model(cfg):
    bg_model, additional_diffuse = sky_model_diffuse(cfg)
    full_sky_model, additional_pts = sky_model_points(cfg, bg_model)
    additional = {**additional_diffuse, **additional_pts}
    return full_sky_model, additional
