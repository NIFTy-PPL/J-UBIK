# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.lax import cond
from jax.scipy.ndimage import map_coordinates


def to_r_phi(cc):
    """
    Transforms form ra-dec (sky) coordinates to r-phi coordinates.
    """
    # FIXME: This assumes that ra is the x-coordinate and dec the y-coordinate
    # and furthermore assumes that the psfs are given in vertical distances
    # (off axis angle). Ensure that this is the correct orientation!
    # VE: The "-jnp.pi/2" come from the fact that the psfset is changing
    # as a function of theta and phi = 0 while phi=0 seems to be measured as the angle from the y axis
    r = jnp.sqrt(cc[..., 0]**2 + cc[..., 1]**2)
    phi = jnp.angle(cc[..., 0] + 1.j*cc[..., 1]) - jnp.pi/2.
    return jnp.stack((r, phi), axis = -1)

# TODO ADD TEST check orthogonality, works, but push the test!
def to_ra_dec(rp):
    """
    Transforms form r-phi coordinates to ra-dec (sky) coordinates.
    """
    x = rp[..., 0]*jnp.cos(rp[..., 1] + jnp.pi/2.)
    y = rp[..., 0]*jnp.sin(rp[..., 1] + jnp.pi/2.)
    return jnp.stack((x,y), axis = -1)


def get_interpolation_weights(rs, r):
    """
    Get bilinear interpolation weights for the psfs along the r-axis. If the 
    requested `r` is outside the given interval, the the psf corresponding to 
    the smallest/largest radius in `rs` is extrapolated.
    """
    dr = rs[1:] - rs[:-1]

    def _get_wgt_front(i):
        res = jnp.zeros(rs.shape, dtype=float)
        res = res.at[0].set(1.)
        return res
    res = cond(r <= rs[0], _get_wgt_front, 
               lambda _: jnp.zeros(rs.shape, dtype=float), 0)

    def _get_wgt_back(i):
        res = jnp.zeros(rs.shape, dtype=float)
        res = res.at[rs.size-1].set(1.)
        return res
    res += cond(r >= rs[rs.size-1], _get_wgt_back,
                lambda _: jnp.zeros(rs.shape, dtype=float), 0)

    def _get_wgt(i):
        res = jnp.zeros(rs.shape, dtype=float)
        wgt = (r - rs[i]) / dr[i]
        res = res.at[i].set(1. - wgt)
        res = res.at[i+1].set(wgt)
        return res
    for i in range(rs.size - 1):
        res += cond((rs[i]<r)*(rs[i+1]>=r), _get_wgt,
                    (lambda _: jnp.zeros(rs.shape, dtype=float)), i)
    return res


def to_patch_coordinates(dcoords, patch_center, patch_delta):
    """
    Transforms distances in sky coordinates to coordinates of the psf patch

    Parameters:
    -----------
    dcoords: numpy.ndarray
        Distances in sky coordinates which shall be transformed
    patch_center: numpy.ndarray
        Indices corresponding to the psf center in the patch.
    patch_delta: numpy.ndarray
        Binsize of the psf patch.
    """
    res = (dcoords + patch_center * patch_delta) / patch_delta
    return jnp.moveaxis(res, -1, 0)


def get_psf(psfs, rs, patch_center_ids, patch_deltas, pointing_center):
    """
    Parameters:
    -----------
    psfs: numpy.ndarray (Shape: rs.shape + psf.shape)
        Psf grids for each off axis radius.
    rs: numpy.ndarray
        1D-Array with off-axis radial coordinates associated with each entry
        of `psfs`.
    patch_center_ids: numpy.ndarray (Shape: rs.shape + (2,))
        Index of the pixel associated with the center of the psf for each entry
        in `psfs`.
    patch_deltas: tuple of float (Shape: (2,))
        Pixelsize of the gridded psfs.
    pointing_center: tuple of float (Shape: (2,))
        Center of the pointing of the module on the sky.
    Returns:
    --------
    func:
        Function that evaluates the psf at ra-dec (sky) coordinates and as a
        function of distance `dra` and `ddec`.
    """
    nrad = rs.size
    if psfs.shape[0] != nrad:
        raise ValueError
    if patch_center_ids.shape[0] != nrad:
        raise ValueError
    rs = np.array(rs, dtype=float)
    # Sort in ascending radii
    sort = np.argsort(rs)
    rs = rs[sort]
    rs = jnp.array(rs)
    psfs = psfs[sort]
    patch_center_ids = patch_center_ids[sort]

    patch_deltas = np.array(list(patch_deltas), dtype=float)
    pointing_center = np.array(list(pointing_center), dtype=float)
    for pp, cc in zip(psfs, patch_center_ids):
        if np.any(cc > np.array(list(pp.shape))):
            raise ValueError
        if np.any(cc < 0):
            raise ValueError

    def psf(ra, dec, dra, ddec):
        """
        ra: right ascession of the center a patch
        dec: declination of the center of a patch
        dra: relative coordinate grid from "the" center (IMHO should be the poiting center)
        ddec: relative coordinate grid from "the" center (sie abaf)
        """
        # Find r and phi corresponding to requested location
        cc = jnp.stack((ra, dec), axis = -1)
        cc -= pointing_center
        rp = to_r_phi(cc)
        # Find and select psfs required for given radius
        shp = rp.shape[:-1]
        get_weights = partial(get_interpolation_weights, rs)
        wgts = vmap(get_weights, in_axes=0, out_axes=0)(rp[..., 0].flatten())
        wgts = wgts.reshape(shp + wgts.shape[-1:])
        wgts = jnp.moveaxis(wgts, -1, 0)

        # Rotate requested psf slice to align with patch
        int_coords = jnp.stack((dra, ddec), axis=-1)
        int_coords = to_r_phi(int_coords)
        int_coords = jnp.moveaxis(int_coords, -1, 0)
        rp = jnp.moveaxis(rp, -1, 0)
        int_coords = int_coords.at[1].set(int_coords[1] - rp[1])
        int_coords = jnp.moveaxis(int_coords, 0, -1)
        int_coords = to_ra_dec(int_coords)

        # Transform psf slice coordinates into patch index coordinates and
        # bilinear interpolate onto slice
        res = jnp.zeros(int_coords.shape[:-1])
        for ids, pp, ww in zip(patch_center_ids, psfs, wgts):
            query_ids = to_patch_coordinates(int_coords, ids, patch_deltas)
            int_res = map_coordinates(pp, query_ids, order=1, mode='nearest')
            res += ww*int_res

        return res
    return psf


def get_psf_func(domain, psf_infos):
    """
    # FIXME Remove domain, is not needed.
    Convenience function for get_psf. Takes a dictionary,
    build by eROSITA-PSF.psf_infos and returns a function.

    Parameters:
    -----------
    psf_infos: dictionary built by #FIXME enter the right name

    returns: psf-function
    """
    psfs = psf_infos['psfs']
    rs = psf_infos['rs']
    patch_center_ids = psf_infos['patch_center_ids']
    patch_deltas = psf_infos['patch_deltas']
    pointing_center = psf_infos['pointing_center']
    return get_psf(psfs, rs, patch_center_ids, patch_deltas, pointing_center)


def psf_interpolator(domain, npatch, psf_infos):
    all_patches = []
    for info in psf_infos:
        # TODO enable this test
        # if not isinstance(info, dict):
        # TypeError("psf_infos has to be a list of dictionaries")
        func_psf = get_psf_func(domain, info)

        shp = (domain.shape[-2], domain.shape[-1])
        dist = (domain.distances[-2], domain.distances[-1])
        for ss in shp:
            if ss % npatch != 0:
                raise ValueError
            if ss % 2 != 0:
                raise ValueError
        patch_shp = tuple(ss//npatch for ss in shp)

        # NOTE This change would symmetrically evaluate the psf but puts
        # the center in between Pixels.
        # c_p = ((np.arange(ss) - ss/2 + 0.5)*dd for ss,dd in zip(shp, dist))
        # centers = (np.array([(i*ss + ss/2 + 0.5)*dd for i in range(npatch)])
        #            for ss, dd in zip(patch_shp, dist))

        # relative coordinates of the PIXEL (CENTERS) corners for all image pixel
        c_p = ((np.arange(ss) - ss/2)*dd for ss, dd in zip(shp, dist))
        # relative coordinates of the patch centers (relative to the upper left corner?)
        centers = (np.array([(i*ss + ss/2)*dd for i in range(npatch)]) for
                   ss, dd in zip(patch_shp, dist))

        c_p = np.meshgrid(*c_p, indexing='ij')
        # FIXME IMHO there should be minus here
        # Flip ra and dec and change sign for ra
        # OLD
        # d_ra = c_p[0]
        # d_ra= c_p[1]
        # NEW
        d_dec = c_p[0]  # NEW
        d_ra = -1.*c_p[1]  # NEW
        #
        # This seems right
        # Using 'xy' here instead of 'ij' ensures correct ordering as requested by
        # OAnew.
        # FIXME this seems to have a different pointing center
        # At least the sign is wrong, and the coordinate center is different to the c_ps
        # The orientation to the edge makes sense when you look at erosita_psf_utils def psf 132
        centers = np.meshgrid(*centers, indexing='xy')
        # OLD
        # c_ra = centers[0].flatten()
        # c_dec = centers[1].flatten()
        # NEW
        c_dec = centers[0].flatten()
        c_ra = jnp.flip(centers[1].flatten())

        patch_psfs = (func_psf(ra, dec, d_ra, d_dec) for ra, dec in
                      zip(c_ra, c_dec))
        patch_psfs = list([np.roll(np.roll(pp, -shp[0]//2, axis=0),
                                   -shp[1]//2,
                                   axis=1)
                           for pp in patch_psfs])
        # patch_psfs = list([pp for pp in patch_psfs]) # FIXME
        # FIXME Patch_psfs should be of shape (n_patches, energies, x, y)
        patch_psfs = np.array(patch_psfs)
        all_patches.append(patch_psfs)
    all_patches = np.stack(all_patches)
    all_patches = np.moveaxis(all_patches, 0, -3)
    return all_patches
