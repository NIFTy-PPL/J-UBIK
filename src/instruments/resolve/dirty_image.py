# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2026 Max-Planck-Society
# Author: Philipp Arras, Andreas Popp

from typing import Union
from jax import linear_transpose

from .data import Observation
from .response import interferometry_response
from .parse import Ducc0Settings, FinufftSettings
from ...grid import Grid
from .constants import RESOLVE_SKY_UNIT, RESOLVE_SPATIAL_UNIT

import jax.numpy as jnp
from astropy.units import Unit
import astropy.units as u
import numpy as np
from numpy.typing import NDArray


def dirty_image(
        observation: Observation,
        sky_grid: Grid,
        backend_settings: Union[Ducc0Settings, FinufftSettings],
        weighting: str = "natural",
        flux_unit: Unit = RESOLVE_SKY_UNIT,
        ) -> u.Quantity:

    # TODO: Extract shape fully from grid
    full_sky_shape = (1, 1, 1) + tuple(sky_grid.spatial.shape)
    
    R = interferometry_response(
        observation = observation,
        sky_grid = sky_grid,
        backend_settings = backend_settings,
    )

    R_adjoint = linear_transpose(R,jnp.ones(full_sky_shape, observation.weight_val.dtype))

    
    d = observation.vis_val
    vol = sky_grid.spatial.dvol.value

    if weighting == "natural":
        w = observation.weight_val
        
    elif weighting == "uniform":
        w = uniform_weights(
            observation = observation,
            sky_grid = sky_grid,
            )    
    else:
        raise ValueError("Either 'natural' or 'uniform' weighting can be chosen.")

    primals = d * w / jnp.sum(w)
    res = R_adjoint(jnp.array(primals))[0] / vol**2 * RESOLVE_SKY_UNIT
    return res.to(flux_unit)

def uniform_weights(
        observation: Observation,
        sky_grid: Grid
        ) -> NDArray:

    weights = np.empty_like(observation.weight_val)
    u, v = observation.effective_uvw()[0:2]

    for ipol in range(observation.npol):
        Hnorm, xedges0, yedges0 = uvw_density(u, v, sky_grid, None)
        H    , xedges , yedges  = uvw_density(u, v, sky_grid, observation.weight_val[ipol])
        assert np.all(xedges == xedges0)
        assert np.all(yedges == yedges0)
        xindices = np.searchsorted(xedges, u.ravel())
        yindices = np.searchsorted(yedges, v.ravel())
        norm = Hnorm[xindices-1, yindices-1].reshape(weights.shape[1:])
        norm *= norm  # FIXME Why
        weights[ipol] = H[xindices-1, yindices-1].reshape(weights.shape[1:]) / norm

        return jnp.array(weights)



def uvw_density(
        eff_u: NDArray, 
        eff_v: NDArray,
        sky_grid: Grid,
        weights: NDArray,
        ) -> Union[NDArray, NDArray, NDArray]:
    
    if weights is not None:
        assert weights.shape == eff_u.shape
        weights = weights.ravel()

    u, v = eff_u.ravel(), eff_v.ravel()

    nx, ny = sky_grid.spatial.shape
    dx, dy = sky_grid.spatial.distances.to(RESOLVE_SPATIAL_UNIT).value

    ku = np.sort(np.fft.fftfreq(nx, dx))
    kv = np.sort(np.fft.fftfreq(ny, dy))

    if np.min(u) < ku[0] or np.max(u)  >= ku[-1] or np.min(v) < kv[0] or np.max(v) >= kv[-1]:
        raise ValueError
    H, xedges, yedges = np.histogram2d(u, v, bins=[ku, kv], weights=weights)
    return H, xedges, yedges
