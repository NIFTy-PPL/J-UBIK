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
from astropy import units as u
from jax import Array, linear_transpose
from jax import numpy as jnp
from jax.tree_util import Partial, tree_map

from ...color import get_2d_binbounds
from ...grid import Grid, PolarizationType
from .constants import RESOLVE_SPATIAL_UNIT, RESOLVE_SPECTRAL_UNIT
from .data.data_modify.frequency import restrict_by_freq
from .data.data_modify.time import restrict_by_time
from .data.observation import Observation
from .parse.response import CufinufftSettings, Ducc0Settings, FinufftSettings
from .util import calculate_phase_offset_to_image_center


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
                [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]]
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


def canonical_sky_to_visibilities(backend_apply, sky_canonical):
    """Apply a raw gridder to a canonical two-dimensional sky slice.

    Canonical arrays have rows increasing North and columns increasing West.
    The raw ducc/FINUFFT backends instead read axes as (l/RA, m/Dec), so this
    boundary adapter transposes once, without conjugation or sign flips.
    With uvw as loaded by ``ms2observations``, a point source obeys
    ``V = vol * exp(+2*pi*i*(u*l_E + v*m_N))``.

    The transpose is C-linear. Its contract is pinned by the radio claim
    and seam tests; the sign-convention history is in
    ``docs/source/user/canonical-sky-design.md``.
    """
    return backend_apply(jnp.transpose(sky_canonical))


def _hermitian_adjoint(response, primals, cotangent):
    """Return ``conj(R^T(conj(cotangent)))`` for a linear response ``R``."""
    transpose = linear_transpose(response, primals)
    conjugate = lambda x: tree_map(jnp.conj, x)
    return conjugate(transpose(conjugate(cotangent))[0])


def interferometry_response(
    observation: Observation,
    sky_grid: Grid,
    backend_settings: Union[Ducc0Settings, FinufftSettings, CufinufftSettings],
):
    """Returns a function computing the radio interferometric response

    Input sky frame is CANONICAL (``dim0 = +Dec``/North, ``dim1 = -RA``/West;
    see ``docs/source/user/canonical-sky-design.md``).  The response owns the conversion to the
    wgridder-native layout: each spatial slice is routed through
    ``canonical_sky_to_visibilities`` before the per-bin backend op, so callers
    author sky cubes in the canonical frame and never transpose themselves.

    Parameters
    ----------
    observation : :class:`resolve.Observation`
        The observation for which the response should compute model
        visibilities.

    sky_domain: SkyDomain
        Providing the information about the discretization of the sky.

    backend_settings: Union[Ducc0Settings, FinufftSettings, CufinufftSettings]
        The backend_settings sets the type of backend, ducc0, finufft or
        cufinufft, which need the following settings:
            - epsilon
            - do_wgridding      (only ducc0)
            - nthreads          (only ducc0)
            - verbosity         (only ducc0)
            - backend           (only ducc0)
            - gpu_maxbatchsize  (only cufinufft)
            - upsampfac         (only cufinufft)
            - dtype             (only cufinufft)
    """
    n_pol = len(sky_grid.polarization)

    # compute bins for time and freq
    n_times = len(sky_grid.times) - 1  # FIXME : This needs to be checked
    bb_times = np.array(sky_grid.times)
    # bb_times = get_binbounds(n_times, sky_domain.times)

    # TODO: Expand logic to discontinuous frequency spacing.
    # frequencies = sky_grid.spectral.binbounds(RESOLVE_SPECTRAL_UNIT).value
    frequencies = get_2d_binbounds(sky_grid.spectral, RESOLVE_SPECTRAL_UNIT)
    n_freqs = len(frequencies)
    # bb_freqs = np.array(frequencies)

    # The sky array is canonical (dim0 = Dec, dim1 = RA); the wgridder x-axis
    # is l/RA, so read the RA quantities from index 1 and Dec from index 0.
    npix_x, npix_y = sky_grid.spatial.shape_xy
    pixsize_x, pixsize_y = sky_grid.spatial.pixel_scales_xy.to(
        RESOLVE_SPATIAL_UNIT
    ).value
    center_x, center_y = calculate_phase_offset_to_image_center(
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
        for freqs in frequencies:
            # TODO: Expand logic to discontinuous frequency spacing.
            ooo, find = restrict_by_freq(oo, freqs[0], freqs[-1], True)
            if any(np.array(ooo.vis.shape) == 0):
                rrr = None
            else:
                if isinstance(backend_settings, Ducc0Settings):
                    rrr = interferometry_response_ducc(
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
                    rrr = interferometry_response_finufft(
                        observation=ooo,
                        pixsize_x=pixsize_x,
                        pixsize_y=pixsize_y,
                        epsilon=backend_settings.epsilon,
                        center_x=center_x,
                        center_y=center_y,
                    )
                elif isinstance(backend_settings, CufinufftSettings):
                    rrr = interferometry_response_cufinufft(
                        observation=ooo,
                        npix_x=npix_x,
                        npix_y=npix_y,
                        pixsize_x=pixsize_x,
                        pixsize_y=pixsize_y,
                        settings=backend_settings,
                        center_x=center_x,
                        center_y=center_y,
                    )
                else:
                    err = (
                        "backend_settings must be an instance of "
                        "`Ducc0Settings`, `FinufftSettings` or "
                        f"`CufinufftSettings`, not {backend_settings}"
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
        if not sky_grid.spectral.is_continuous:
            raise RuntimeError(
                "During response instantiation, the frequency shape of the visibilities"
                "didn't match the sky.\n"
                "Consider using : rve.restrict_to_discontinuous_frequencies"
            )

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
                    r = canonical_sky_to_visibilities(op, inp)
                    res = res.at[pp, row_indices[tt][ff], freq_indices[tt][ff]].set(r)
        return convert_polarization(res, inp_pol, out_pol)

    return apply_R


def interferometry_response_ducc(
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


def _uv_radians_and_phase_shift(observation, pixsize_x, pixsize_y, center_x, center_y):
    """NUFFT coordinates of the visibilities and the phase shift to the image centre.

    Baseline coordinates in metres times frequency over c give the baseline
    in wavelengths; times 2 pi and the pixel size this becomes the NUFFT
    coordinate in radians, wrapped to ``[0, 2 pi)``. ``u`` runs along axis 0
    of the sky and ``v`` enters with a minus sign. When the image centre and
    the phase centre differ, each visibility picks up a complex phase factor,
    returned here so the caller can multiply it onto the NUFFT output. Shared
    by the finufft and cufinufft backends.

    Parameters
    ----------
    observation : Observation
        Provides ``uvw`` in metres, shape ``(n_rows, 3)``, and ``freq`` in Hz.
    pixsize_x, pixsize_y : float
        Pixel size of the sky along axes 0 and 1, in radians.
    center_x, center_y : float
        Offset between the image centre and the phase centre in radians, as
        returned by ``calculate_phase_offset_to_image_center``. It enters the
        phase as direction cosines ``l, m``, which is exact for small offsets.

    Returns
    -------
    u_finu, v_finu : numpy.ndarray, shape ``(n_rows * n_freqs,)``
        NUFFT coordinates, ordered row major over ``(row, frequency)``.
    phase_shift : jax.Array or None
        Complex factor per visibility, same shape and order, or None when no
        shift is applied.
    """
    freq = observation.freq
    uvw = observation.uvw
    speedoflight = 299792458.0

    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    u, v, w = uvw.T

    u_finu = (2 * np.pi * u * pixsize_x) % (2 * np.pi)
    v_finu = (-2 * np.pi * v * pixsize_y) % (2 * np.pi)

    if ((center_x is not None) and (center_y is not None)) or (
        center_x != 0.0 and center_y != 0.0
    ):
        n = np.sqrt(1 - center_x**2 - center_y**2)
        phase_shift = np.exp(-2j * np.pi * (u * center_x + v * center_y + w * (n - 1)))
        phase_shift = jnp.array(phase_shift)
    else:
        phase_shift = None
    return u_finu, v_finu, phase_shift


class CufinufftResponse:
    """Radio response on the GPU through persistent cufinufft plans.

    Maps a sky image to model visibilities, like the finufft backend, but
    holds a :class:`~jubik.instruments.resolve.cufinufft.PlanSet` for the
    visibility coordinates so that plan setup and point sorting are paid once
    per geometry instead of on every call. The instance is a plain callable
    and can be used inside ``jit``, ``grad`` and ``vmap``.

    Parameters
    ----------
    observation : Observation
        Provides ``uvw`` and ``freq`` of the visibilities.
    npix_x, npix_y : int
        Sky shape along axes 0 and 1.
    pixsize_x, pixsize_y : float
        Pixel size along axes 0 and 1, in radians.
    settings : CufinufftSettings
        Accuracy, precision and memory options of the transform.
    center_x, center_y : float
        Offset between the image centre and the phase centre in radians, see
        :func:`_uv_radians_and_phase_shift`.

    Attributes
    ----------
    plans : PlanSet
        The plans of this response. Compiled likelihoods keep the PlanSet
        alive on their own, so dropping this instance does not free the plans.
        Call ``plans.close()`` to release the GPU resources early if wanted.
    """

    def __init__(
        self,
        observation,
        npix_x,
        npix_y,
        pixsize_x,
        pixsize_y,
        settings: CufinufftSettings,
        center_x=None,
        center_y=None,
    ):
        """Compute the NUFFT coordinates and upload them into a PlanSet.

        No plan is built yet; that happens when a program calling this
        response is compiled.
        """
        from .cufinufft import PlanSet

        self._n_freqs = len(observation.freq)
        self._vol = pixsize_x * pixsize_y
        u_finu, v_finu, self._phase_shift = _uv_radians_and_phase_shift(
            observation, pixsize_x, pixsize_y, center_x, center_y
        )
        self.plans = PlanSet(
            (npix_x, npix_y),
            u_finu,
            v_finu,
            eps=settings.epsilon,
            dtype=np.dtype(settings.dtype),
            gpu_maxbatchsize=settings.gpu_maxbatchsize,
            upsampfac=settings.upsampfac,
        )

    def __call__(self, inp):
        """Map a sky image to model visibilities.

        Computes ``vol * nufft2(sky) * phase``, where ``vol`` is the pixel area
        and ``phase`` the per visibility phase shift, then reshapes the flat
        result to one column per frequency.

        Parameters
        ----------
        inp : array, shape ``(npix_x, npix_y)``
            Sky brightness per pixel, cast to the precision of the settings.

        Returns
        -------
        array, shape ``(n_rows, n_freqs)``
            Complex model visibilities.
        """
        from .cufinufft import nufft2

        res = self._vol * nufft2(inp, self.plans)
        if self._phase_shift is not None:
            res = res * self._phase_shift
        return res.reshape(-1, self._n_freqs)


def interferometry_response_cufinufft(
    observation,
    npix_x,
    npix_y,
    pixsize_x,
    pixsize_y,
    settings: CufinufftSettings,
    center_x=None,
    center_y=None,
):
    """Build the cufinufft radio response, see :class:`CufinufftResponse`.

    Functional entry point matching the other backends, used by
    :func:`interferometry_response` when given :class:`CufinufftSettings`.
    Parameters are those of :class:`CufinufftResponse`.

    Returns
    -------
    CufinufftResponse
        Callable mapping a sky of shape ``(npix_x, npix_y)`` to visibilities
        of shape ``(n_rows, n_freqs)``.
    """
    return CufinufftResponse(
        observation=observation,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        settings=settings,
        center_x=center_x,
        center_y=center_y,
    )


def interferometry_response_finufft(
    observation, pixsize_x, pixsize_y, epsilon, center_x=None, center_y=None
):
    from jax_finufft import nufft2

    freq = observation.freq
    vol = pixsize_x * pixsize_y
    u_finu, v_finu, phase_shift = _uv_radians_and_phase_shift(
        observation, pixsize_x, pixsize_y, center_x, center_y
    )

    def apply_finufft(inp, u, v, eps):
        res = vol * nufft2(inp.astype(np.complex128), u, v, eps=eps)
        if phase_shift is not None:
            res = res * phase_shift
        return res.reshape(-1, len(freq))

    R = Partial(apply_finufft, u=u_finu, v=v_finu, eps=epsilon)
    return R
