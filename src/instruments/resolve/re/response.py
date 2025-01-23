import numpy as np
from jax import numpy as jnp

from .parse.response import SkyDomain, Ducc0Settings, FinufftSettings
from ..data.observation import Observation
from jax.tree_util import Partial


from typing import Union


def dtype_float2complex(dt):
    if dt == np.float64:
        return np.complex128
    if dt == np.float32:
        return np.complex64
    raise ValueError


def get_binbounds(size, coordinates):
    if len(coordinates) == 1:
        return np.array([-np.inf, np.inf])
    c = np.array(coordinates)
    bounds = np.empty(size + 1)
    bounds[1:-1] = c[:-1] + 0.5 * np.diff(c)
    bounds[0] = c[0] - 0.5 * (c[1] - c[0])
    bounds[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return bounds


def convert_polarization(inp, inp_pol, out_pol):
    if inp_pol == ("I", "Q", "U", "V"):
        if out_pol == ("RR", "RL", "LR", "LL"):
            mat_stokes_to_circular = jnp.array(
                [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]]
            )
            return jnp.tensordot(mat_stokes_to_circular, inp, axes=([0], [0]))
        elif out_pol == ("XX", "XY", "YX", "YY"):
            mat_stokes_to_linear = jnp.array(
                [[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1j, -1j]]
            )
            return jnp.tensordot(mat_stokes_to_linear, inp, axes=([0], [0]))
    elif inp_pol == ("I",):
        if out_pol == ("LL", "RR") or out_pol == ("XX", "YY"):
            new_shp = list(inp.shape)
            new_shp[0] = 2
            return jnp.broadcast_to(inp, new_shp)
        if len(out_pol) == 1 and out_pol[0] in ("I", "RR", "LL", "XX", "YY"):
            return inp
    err = f"conversion of polarization {inp_pol} to {out_pol} not implemented. Please implement!"
    raise NotImplementedError(err)


def InterferometryResponse(
    observation: Observation,
    sky_domain: SkyDomain,
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
    n_pol = len(sky_domain.polarization_labels)

    # compute bins for time and freq
    n_times = len(sky_domain.times)
    bb_times = get_binbounds(n_times, sky_domain.times)

    n_freqs = len(sky_domain.frequencies)
    bb_freqs = get_binbounds(n_freqs, sky_domain.frequencies)

    # build responses for: time binds, freq bins
    sr = []
    row_indices, freq_indices = [], []
    for t in range(n_times):
        sr_tmp, t_tmp, f_tmp = [], [], []
        if tuple(bb_times[t: t + 2]) == (-np.inf, np.inf):
            oo = observation
            tind = slice(None)
        else:
            oo, tind = observation.restrict_by_time(
                bb_times[t], bb_times[t + 1], True)
        for f in range(n_freqs):
            ooo, find = oo.restrict_by_freq(bb_freqs[f], bb_freqs[f + 1], True)
            if any(np.array(ooo.vis.shape) == 0):
                rrr = None
            else:
                if isinstance(backend_settings, Ducc0Settings):
                    rrr = InterferometryResponseDucc(
                        observation=ooo,
                        npix_x=sky_domain.npix_x,
                        npix_y=sky_domain.npix_y,
                        pixsize_x=sky_domain.pixsize_x,
                        pixsize_y=sky_domain.pixsize_y,
                        do_wgridding=backend_settings.do_wgridding,
                        epsilon=backend_settings.epsilon,
                        nthreads=backend_settings.nthreads,
                        verbosity=backend_settings.verbosity,
                        center_x=sky_domain.center_x,
                        center_y=sky_domain.center_y,
                    )
                elif isinstance(backend_settings, FinufftSettings):
                    print('Using Finufft')
                    rrr = InterferometryResponseFinuFFT(
                        observation=ooo,
                        pixsize_x=sky_domain.pixsize_x,
                        pixsize_y=sky_domain.pixsize_y,
                        epsilon=backend_settings.epsilon,
                        center_x=sky_domain.center_x,
                        center_y=sky_domain.center_y,
                    )
                else:
                    err = ("backend_settings must be an instance of "
                           "`Ducc0Settings` or `FinufftSettings`, not "
                           f"{backend_settings}")
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

    inp_pol = tuple(sky_domain.polarization_labels)
    out_pol = observation.vis.domain[0].labels

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
                    res = res.at[pp, row_indices[tt][ff],
                                 freq_indices[tt][ff]].set(r)
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
    observation,
    pixsize_x,
    pixsize_y,
    epsilon,
    center_x=None,
    center_y=None
):
    from jax_finufft import nufft2

    freq = observation.freq
    uvw = observation.uvw
    vol = pixsize_x * pixsize_y
    speedoflight = 299792458.0

    uvw = np.transpose(
        (uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    u, v, w = uvw.T

    u_finu = (2 * np.pi * u * pixsize_x) % (2 * np.pi)
    v_finu = (-2 * np.pi * v * pixsize_y) % (2 * np.pi)

    if (center_x is not None) and (center_y is not None):
        n = np.sqrt(1 - center_x**2 - center_y**2)
        phase_shift = np.exp(-2j*np.pi * (u*center_x + v*center_y + w*(n-1)))
        phase_shift = jnp.array(phase_shift)
    else:
        phase_shift = 1

    def apply_finufft(inp, u, v, eps):
        res = vol * nufft2(inp.astype(np.complex128), u, v, eps=eps)
        res = res * phase_shift
        return res.reshape(-1, len(freq))

    R = Partial(apply_finufft, u=u_finu, v=v_finu, eps=epsilon)
    return R
