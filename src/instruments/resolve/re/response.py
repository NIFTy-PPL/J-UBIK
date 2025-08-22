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
# Copyright(C) 2025 Max-Planck-Society
# Author: Jakob Roth, Julian Rüstig

from typing import Union

import numpy as np
from jax.tree_util import Partial
from jax import Array, numpy as jnp
from astropy import units as u

from ..parse.response import Ducc0Settings, FinufftSettings
from ..data.observation import Observation
from ..data.data_modify.frequency_handling import restrict_by_freq
from ..data.data_modify.time_modify import restrict_by_time
from ....grid import Grid, PolarizationType
from ..util import calculate_phase_offset_to_image_center


SPECTRAL_UNIT = u.Hz
SPATIAL_UNIT = u.rad


def dtype_float2complex(dt):
    if dt == np.float64:
        return np.complex128
    if dt == np.float32:
        return np.complex64
    raise ValueError


def get_binbounds(size, coordinates):
    if len(coordinates) == 1:
        return np.array([-np.inf, np.inf])
    coords = np.array(coordinates)
    bounds = np.empty(size + 1)
    bounds[1:-1] = coords[:-1] + 0.5 * np.diff(coords)
    bounds[0] = coords[0] - 0.5 * (coords[1] - coords[0])
    bounds[-1] = coords[-1] + 0.5 * (coords[-1] - coords[-2])
    return bounds


def convert_polarization(
    inp: Array, inp_pol: PolarizationType, out_pol: PolarizationType
):
    if inp_pol == PolarizationType(("I", "Q", "U", "V")):
        if out_pol == PolarizationType(("RR", "RL", "LR", "LL")):
            mat_stokes_to_circular = jnp.array(
                [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]]
            )
            return jnp.tensordot(mat_stokes_to_circular, inp, axes=([0], [0]))

        elif out_pol == PolarizationType(("XX", "XY", "YX", "YY")):
            mat_stokes_to_linear = jnp.array(
                [[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1j, -1j]]
            )
            return jnp.tensordot(mat_stokes_to_linear, inp, axes=([0], [0]))

    elif inp_pol == PolarizationType(("I",)):
        if out_pol == PolarizationType(("LL", "RR")) or out_pol == PolarizationType(
            ("XX", "YY")
        ):
            new_shp = list(inp.shape)
            new_shp[0] = 2
            return jnp.broadcast_to(inp, new_shp)
        if out_pol.is_single_feed:
            return inp
    err = f"conversion of polarization {inp_pol} to {out_pol} not implemented. Please implement!"
    raise NotImplementedError(err)


def InterferometryResponse(
    observation: Observation,
    sky_grid: Grid,
    backend_settings: Union[Ducc0Settings, FinufftSettings],
):
    """Returns a function computing the radio interferometric response

    Parameters
    ----------
    observation : :class:`resolve.Observation`
        The observation for which the response should compute model
        visibilities.

    sky_domain: SkyDomain
        Providing the information about the discretization of the sky.

    backend_settings: Union[Ducc0Settings, FinufftSettings]
        The backend_settings sets the type of backend, either ducc0 or finufft,
        which need the following settings:
            - epsilon
            - do_wgridding  (only ducc0)
            - nthreads      (only ducc0)
            - verbosity     (only ducc0)
            - backend       (only ducc0)
    """
    n_pol = len(sky_grid.polarization)

    # compute bins for time and freq
    n_times = len(sky_grid.times) - 1  # FIXME : This needs to be checked
    bb_times = np.array(sky_grid.times)
    # bb_times = get_binbounds(n_times, sky_domain.times)

    frequencies = sky_grid.spectral.binbounds_in(SPECTRAL_UNIT)
    n_freqs = len(frequencies) - 1
    bb_freqs = np.array(frequencies)
    # bb_freqs = get_binbounds(n_freqs, sky_domain.frequencies)

    npix_x, npix_y = sky_grid.spatial.shape
    pixsize_x, pixsize_y = sky_grid.spatial.distances_in(SPATIAL_UNIT)
    center_y, center_x = calculate_phase_offset_to_image_center(
        sky_grid.spatial.center,
        sky_grid.spatial.center
        if observation.direction is None
        else observation.direction.to_sky_coord(),
    )

    # build responses for: time binds, freq bins
    sr = []
    row_indices, freq_indices = [], []
    for t in range(n_times):
        sr_tmp, t_tmp, f_tmp = [], [], []
        if tuple(bb_times[t : t + 2]) == (-np.inf, np.inf):
            oo = observation
            tind = slice(None)
        else:
            oo, tind = restrict_by_time(observation, bb_times[t], bb_times[t + 1], True)
        for f in range(n_freqs):
            ooo, find = restrict_by_freq(oo, bb_freqs[f], bb_freqs[f + 1], True)
            if any(np.array(ooo.vis.shape) == 0):
                rrr = None
            else:
                if isinstance(backend_settings, Ducc0Settings):
                    rrr = InterferometryResponseDucc(
                        observation=ooo,
                        npix_x=npix_x,
                        npix_y=npix_y,
                        pixsize_x=pixsize_x,
                        pixsize_y=pixsize_y,
                        do_wgridding=backend_settings.do_wgridding,
                        epsilon=backend_settings.epsilon,
                        nthreads=backend_settings.nthreads,
                        verbosity=backend_settings.verbosity,
                        center_x=center_x,
                        center_y=center_y,
                    )
                elif isinstance(backend_settings, FinufftSettings):
                    rrr = InterferometryResponseFinuFFT(
                        observation=ooo,
                        pixsize_x=pixsize_x,
                        pixsize_y=pixsize_y,
                        epsilon=backend_settings.epsilon,
                        center_x=center_x,
                        center_y=center_y,
                    )
                else:
                    err = (
                        "backend_settings must be an instance of "
                        "`Ducc0Settings` or `FinufftSettings`, not "
                        f"{backend_settings}"
                    )
                    raise ValueError(err)

            sr_tmp.append(rrr)
            t_tmp.append(tind)
            f_tmp.append(find)
        sr.append(sr_tmp)
        row_indices.append(t_tmp)
        freq_indices.append(f_tmp)

    target_shape = (n_pol,) + tuple(observation.vis.shape[1:])
    foo = np.zeros(target_shape, np.int8)
    for pp in range(n_pol):
        for tt in range(n_times):
            for ff in range(n_freqs):
                foo[pp, row_indices[tt][ff], freq_indices[tt][ff]] = 1.0
    if np.any(foo == 0):
        raise RuntimeError("This should not happen. Please report.")

    inp_pol = sky_grid.polarization
    out_pol = observation.polarization

    def apply_R(sky):
        res = jnp.empty(target_shape, dtype_float2complex(sky.dtype))
        for pp in range(sky.shape[0]):
            for tt in range(sky.shape[1]):
                for ff in range(sky.shape[2]):
                    op = sr[tt][ff]
                    if op is None:
                        continue
                    inp = sky[pp, tt, ff]
                    r = op(inp)
                    res = res.at[pp, row_indices[tt][ff], freq_indices[tt][ff]].set(r)
        return convert_polarization(res, inp_pol, out_pol)

    return apply_R


def InterferometryResponseDucc(
    observation,
    npix_x,
    npix_y,
    pixsize_x,
    pixsize_y,
    do_wgridding,
    epsilon,
    nthreads=1,
    verbosity=1,
    **kwargs,
):
    from jaxbind.contrib import jaxducc0

    vol = pixsize_x * pixsize_y

    wg = jaxducc0.get_wgridder(
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        npix_x=npix_x,
        npix_y=npix_y,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
        verbosity=verbosity,
        flip_v=True,
        **kwargs,
    )
    wgridder = Partial(wg, observation.uvw, observation.freq)

    return lambda x: vol * wgridder(x)[0]


def InterferometryResponseFinuFFT(
    observation, pixsize_x, pixsize_y, epsilon, center_x=None, center_y=None
):
    from jax_finufft import nufft2

    freq = observation.freq
    uvw = observation.uvw
    vol = pixsize_x * pixsize_y
    speedoflight = 299792458.0

    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    u, v, w = uvw.T

    u_finu = (2 * np.pi * u * pixsize_x) % (2 * np.pi)
    v_finu = (-2 * np.pi * v * pixsize_y) % (2 * np.pi)

    if (center_x is not None) and (center_y is not None):
        n = np.sqrt(1 - center_x**2 - center_y**2)
        phase_shift = np.exp(-2j * np.pi * (u * center_x + v * center_y + w * (n - 1)))
        phase_shift = jnp.array(phase_shift)
    else:
        phase_shift = 1

    def apply_finufft(inp, u, v, eps):
        res = vol * nufft2(inp.astype(np.complex128), u, v, eps=eps)
        res = res * phase_shift
        return res.reshape(-1, len(freq))

    R = Partial(apply_finufft, u=u_finu, v=v_finu, eps=epsilon)
    return R
