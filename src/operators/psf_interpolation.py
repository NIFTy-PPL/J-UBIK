import nifty8 as ift
import numpy as np
import jax
import jax.numpy as jnp

from jax.scipy.ndimage import map_coordinates
from .convolution_operators import OAnew
from .adg.nifty_convolve import get_convolve

def to_r_phi(cc):
    """
    Transforms form ra-dec (sky) coordinates to r-phi coordinates.
    """
    r = jnp.sqrt(cc[0]**2 + cc[1]**2)
    phi = jnp.angle(cc[0] + 1.j*cc[1])
    return jnp.array([r, phi])

def to_ra_dec(rp):
    """
    Transforms form r-phi coordinates to ra-dec (sky) coordinates.
    """
    x, y = rp[0]*jnp.cos(rp[1]), rp[0]*jnp.sin(rp[1])
    return jnp.array([x, y])

def get_interpolation_weights(rs, r):
    dr = rs[1:] - rs[:-1]

    def _get_wgt_front(i):
        res = jnp.zeros_like(rs)
        res = res.at[0].set(1.)
        return res
    res = jax.lax.cond(r < rs[0], _get_wgt_front, lambda _: jnp.zeros_like(rs), 0)

    def _get_wgt_back(i):
        res = jnp.zeros_like(rs)
        res = res.at[rs.size-1].set(1.)
        return res    
    res += jax.lax.cond(r >= rs[rs.size-1], _get_wgt_back, lambda _: jnp.zeros_like(rs), 0)

    def _get_wgt(i):
        res = jnp.zeros_like(rs)
        wgt = (r - rs[i]) / dr[i]
        res = res.at[i].set(1. - wgt)
        res = res.at[i+1].set(wgt)
        return res
    for i in range(rs.size - 1):
        res += jax.lax.cond((rs[i]<r)*(rs[i+1]>=r), 
                _get_wgt, (lambda _: jnp.zeros_like(rs)), i)
    return res


def find_interpolate_index_r(rs, r):
    """
    Find indices and weights for bilinear interpolation of the psfs
    along the r-axis. If the requested `r` is outside the given
    interval, the index of the psf corresponding to the
    smallest/largest radius in `rs` is returned.
    """
    upper = rs[1:]
    lower = rs[:-1]
    i_upper = []
    i_lower = []
    for i in range(upper.size):
        i_upper += jax.lax.cond(upper[i] < r, (lambda i: [i,]), (lambda _: [-1]), i)
        i_lower += jax.lax.cond(lower[i] >=r, (lambda i: [i,]), (lambda _: [-1]), i)
    i_upper = jnp.array(i_upper)
    i_upper = i_upper[i_upper != -1]
    
    #i_upper = jnp.where(upper < r)[0]
    #i_lower = jnp.where(lower >= r)[0]
    # If `r` larger then biggest radius extrapolate psf
    if i_upper.size == upper.size:
        return (rs.size-1,), (1.,)
    # If `r` smaller then smallest radius extrapolate psf
    if i_lower.size == lower.size:
        return (0,), (1.,)
    # Select psfs and bilinear interpolate
    i_all = jnp.arange(rs.size - 1, dtype=int)
    i_all = i_all[~jnp.isin(i_all, i_upper)]
    i_all = i_all[~jnp.isin(i_all, i_lower)]
    assert i_all.size == 1
    i_all = i_all[0]
    th0, th1 = rs[i_all], rs[i_all+1]
    dth = th1 - th0
    wgt = (r - th0) / dth
    return (i_all, i_all + 1), ((1. - wgt), wgt)

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
    tm = patch_center*patch_delta
    res = jnp.swapaxes(dcoords, 0, -1) + tm
    res /= patch_delta
    return jnp.swapaxes(res, -1, 0)

def nn_interpol(radec, grid_radec, vals):
    pdist = ((radec - grid_radec)** 2).sum(axis=-1)
    return jnp.mean(vals[pdist == jnp.min(pdist)])

def get_psf(psfs, rs, patch_center_ids, patch_deltas, pointing_center, 
            radec_limits):
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
        Center of the pointing of the module on the sky. Must be within 
        `radec_limits`.
    radec_limits: tuple of tuple of float
        Lower and upper limit of the sky coordinate frame in ra-dec units
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
    # Sort in ascending radii
    sort = np.argsort(rs)
    rs = rs[sort]
    rs = jnp.array(rs)
    psfs = psfs[sort]
    patch_center_ids = patch_center_ids[sort]

    patch_deltas = np.array(list(patch_deltas))
    pointing_center = np.array(list(pointing_center))
    lower = np.array(list(radec_limits[0]))
    upper = np.array(list(radec_limits[1]))
    if np.any(upper <= lower):
        raise ValueError
    if np.any(pointing_center < lower):
        raise ValueError
    if np.any(pointing_center > upper):
        raise ValueError
    pointing_center -= lower
    for pp, cc in zip(psfs, patch_center_ids):
        if np.any(cc > np.array(list(pp.shape))):
            raise ValueError
        if np.any(cc < 0):
            raise ValueError

    def psf(ra, dec, dra, ddec):
        # Find r and phi corresponding to requested location
        cc = jnp.array([ra, dec])
        cc -= pointing_center
        rp = to_r_phi(cc)
        # Find and select psfs required for given radius
        #inds, wgts = find_interpolate_index_r(rs, rp[0])
        wgts = get_interpolation_weights(rs, rp[0])

        # Rotate requested psf slice to align with patch
        int_coords = jnp.stack((dra, ddec), axis = 0)
        int_coords = to_r_phi(int_coords)
        int_coords = int_coords.at[1].set(int_coords[1] - rp[1])
        int_coords = to_ra_dec(int_coords)

        # Transform psf slice coordinates into patch index coordinates and
        # bilinear interpolate onto slice
        res = jnp.zeros(int_coords.shape[1:])
        for ids, pp, ww in zip(patch_center_ids, psfs, wgts):
            query_ids = to_patch_coordinates(int_coords, ids, patch_deltas)
            int_res = map_coordinates(pp, query_ids, order=1, mode='nearest')
            res += ww*int_res

        return res

    return psf

def get_psf_func(domain, lower_radec, obs_infos):
    psfs = obs_infos['psfs']
    rs = obs_infos['rs']
    patch_center_ids = obs_infos['patch_center_ids']
    patch_deltas = obs_infos['patch_deltas']
    pointing_center = obs_infos['pointing_center']

    if not isinstance(domain, ift.RGSpace):
        raise ValueError
    if not domain.harmonic == False:
        raise ValueError

    upper_radec = (tuple(ll+ss*dd for ll,ss,dd in 
                   zip(lower_radec, domain.shape, domain.distances)))
    radec_limits = (lower_radec, upper_radec)
    return get_psf(psfs, rs, patch_center_ids, patch_deltas, 
                   pointing_center, radec_limits)

def psf_convolve_operator(domain, lower_radec, obs_infos, msc_infos):
    # NOTE: Assumes the repository "https://gitlab.mpcdf.mpg.de/pfrank/adg.git"
    # to be cloned and located in a folder named "adg" within the folder of this
    # python file.
    
    c = msc_infos['c']
    q = msc_infos['q']
    b = msc_infos['b']
    min_m0 = msc_infos['min_m0']
    linear = msc_infos['linear']
    local = True

    func_psf = get_psf_func(domain, lower_radec, obs_infos)
    return get_convolve(domain, func_psf, c, q, b, min_m0, linear, local)

def psf_lin_int_operator(domain, npatch, lower_radec, obs_infos, margfrac=0.1):
    func_psf = get_psf_func(domain, lower_radec, obs_infos)

    shp = domain.shape
    dist = domain.distances
    for ss in shp:
        if ss%npatch != 0:
            raise ValueError
        if ss%2 != 0:
            raise ValueError
    patch_shp = tuple(ss//npatch for ss in shp)
    c_p = ((np.arange(ss) - ss/2 + 0.5)*dd for ss,dd in zip(shp, dist))
    c_p = np.meshgrid(*c_p, indexing='ij')
    d_ra = c_p[0]
    d_dec = c_p[1]

    centers = (np.array([(i*ss + ss/2 + 0.5)*dd for i in range(npatch)]) for 
               ss, dd in zip(patch_shp, dist))
    centers = np.meshgrid(*centers, indexing='ij')
    c_ra = centers[0].flatten()
    c_dec = centers[1].flatten()

    patch_psfs = (func_psf(ra, dec, d_ra, d_dec) for ra, dec in 
                       zip(c_ra, c_dec))
    patch_psfs = list(
        [np.roll(np.roll(pp, -shp[0]//2, axis = 0), -shp[1]//2, axis = 1)
        for pp in patch_psfs])
    patch_psfs = np.array(patch_psfs)
    margin = max((int(np.ceil(margfrac*ss)) for ss in patch_shp))
    op = OAnew.force(domain, patch_psfs, len(patch_psfs), margin)
    return op
