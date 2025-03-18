# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Jakob Roth

from .resolve_point_sources import resolve_point_sources
from .resolve_diffuse import sky_model_diffuse
from ..parse.sky_model.resolve_point_sources import ResolvePointSourcesModel
from ..parse.sky_model.resolve_diffuse import ResolveDiffuseSkyModel

import nifty8 as ift

import numpy as np


# TODO: replace with new config parsing!
ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
SPEEDOFLIGHT = 299792458.0
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


def _spatial_dom(sky_cfg):
    nx = sky_cfg.getint("space npix x")
    ny = sky_cfg.getint("space npix y")
    dx = str2rad(sky_cfg["space fov x"]) / nx
    dy = str2rad(sky_cfg["space fov y"]) / ny
    return ift.RGSpace([nx, ny], [dx, dy])


def sky_model(sky_cfg):
    sky_dom = _spatial_dom(sky_cfg)

    bg_model, additional_diffuse = sky_model_diffuse(
        sky_dom.shape,
        sky_dom.distances,
        ResolveDiffuseSkyModel.from_config_parser(sky_cfg),
    )
    full_sky_model, additional_pts = resolve_point_sources(
        sky_dom,
        ResolvePointSourcesModel.cfg_to_resolve_point_sources(sky_cfg),
        bg=bg_model,
    )

    additional = {**additional_diffuse, **additional_pts}
    return full_sky_model, additional
