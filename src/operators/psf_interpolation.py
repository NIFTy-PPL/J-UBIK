import numpy as np
import jax.numpy as jnp
from jax import vmap

def to_r_phi(cc):
    """
    Transforms form ra-dec (sky) coordinates to r-phi coordinates.
    """
    r = jnp.sqrt(cc[0]**2 + cc[1]**2)
    phi = jnp.angle(cc[0] + 1.j*cc[1])
    return jnp.array([r, phi])

def to_ra_dec(rp):
    x, y = rp[0]*jnp.cos(rp[1]), rp[0]*jnp.sin(rp[1])
    return jnp.array([x, y])

def find_interpolate_index_r(rs, r):
    """
    Find indices and weights for bilinear interpolation of the psfs
    along the r-axis. If the requested `r` is outside the given
    interval, the index of the psf corresponding to the
    smallest/largest radius in `rs` is returned.
    """
    upper = rs[1:]
    lower = rs[:-1]
    i_upper = jnp.where(upper < r)[0]
    i_lower = jnp.where(lower >= r)[0]
    # If `r` larger then biggest radius extrapolate psf
    if i_upper.size == upper.size:
        return [rs.size-1,], [1.,]
    # If `r` smaller then smallest radius extrapolate psf
    if i_lower.size == lower.size:
        return [0,], [1.,]
    # Select psfs and bilinear interpolate
    i_all = jnp.arange(rs.size - 1, dtype=int)
    i_all = i_all[~jnp.isin(i_all, i_upper)]
    i_all = i_all[~jnp.isin(i_all, i_lower)]
    assert i_all.size == 1
    i_all = i_all[0]
    th0, th1 = rs[i_all], rs[i_all+1]
    dth = th1 - th0
    wgt = (r - th0) / dth
    return [i_all, i_all + 1], [(1. - wgt), wgt]

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
    psfs = psfs[sort]
    patch_center_ids = patch_center_ids[sort]

    pointing_center = np.array(list(pointing_center))
    lower = np.array(list(radec_limits[0]))
    upper = np.array(list(radec_limits[1]))
    if np.any(upper <= lower):
        raise ValueError
    pointing_center -= lower
    window_size = upper - lower



    def psf(ra, dec, dra, ddec):
        cc = jnp.array([ra, dec])
        cc -= pointing_center
        rp = to_r_phi(cc)
        i_psf, wgt = bilinear_interpolate_psf_r(psfs, rs, rp[0])



        #dcc = jnp.stack((dra, ddec), axis = -1)
        r, phi = to_r_phi(ra, dec, center)
        mypsf = bilinear_interpolate_psf_r(r_psfs, rs, r)
        psfra = dradecs[:,:,0] + center[0]
        psfdec = dradecs[:,:,1] + center[1]
        psfr, psfphi = to_r_phi(psfra, psfdec, center)
        psfphi += phi
        psfphi = psfphi%(2.*jnp.pi)
        psfra, psfdec = to_ra_dec(psfr, psfphi, center)
        psfra, psfdec = psfra - center[0], psfdec - center[1]
        psfra, psfdec = psfra.flatten(), psfdec.flatten()
        mypsf = mypsf.flatten()

        myra, mydec = dra.flatten(), ddec.flatten()

        myradec = jnp.stack((myra, mydec), axis = -1)
        psfradec = jnp.stack((psfra, psfdec), axis = -1)
        respsf = []
        #TODO use scipy.ndimage.map_coordinates
        for i, vv in enumerate(myradec):
            respsf.append(nn_interpol(vv, psfradec, mypsf))
        respsf = jnp.array(respsf)
        return respsf.reshape(dra.shape)

    return psf

def test_psf():
    import pylab as plt

    max_radec = (2.,2.)
    npix_x, npix_y = 128, 128

    ra = np.arange(npix_x) / npix_x * max_radec[0]
    ra -= 0.5*max_radec[0]
    dec = np.arange(npix_y) / npix_y * max_radec[1]
    dec -= 0.5*max_radec[1]
    ra, dec = ra[1:], dec[1:]
    print(ra)
    print(dec)


    center = (1.,1.)

    sig = 0.1
    def func(r, dx,dy):
        dr = np.sqrt((dx/(1.+2.*r**2))**2 + dy**2)
        return np.exp(-0.5*(dr/sig)**2)

    rs = np.array([0., 0.1, 0.5, 0.7, 1.])

    nx = 128*2
    ny = 128*2
    dra = np.linspace(-1.,1., num = nx)
    ddec = np.linspace(-1.,1., num = ny)
    dra, ddec = np.meshgrid(dra, ddec, indexing='ij')
    dradecs = np.stack((dra, ddec), axis = -1)

    psfs = list([func(rr, dra, ddec) for rr in rs])
    psfs = np.stack(psfs, axis = 0)

    for pp,rr in zip(psfs,rs):
        plt.imshow(pp.T, origin='lower')
        plt.title(f'radius = {rr}')
        plt.show()

    func_psf = get_psf(rs, dradecs, psfs, center, max_radec)

    ddra, dddec = jnp.meshgrid(ra, dec, indexing='ij')
    ra, dec = .5, .2
    mypsf = func_psf(ra, dec, ddra, dddec)
    im = plt.imshow(mypsf.T, origin='lower', vmin = 0., vmax = 1.)
    plt.colorbar(im)
    plt.show()

    test_r = np.sqrt((ra-center[0])**2 + (dec-center[1])**2)
    gtpsf = func(test_r, ddra, dddec)
    im = plt.imshow(gtpsf.T, origin='lower', vmin = 0., vmax = 1.)
    plt.colorbar(im)
    plt.show()


test_psf()