import nifty8 as ift
import numpy as np
import jax.numpy as jnp

from ducc0.fft import good_size as good_fft_size
from functools import partial
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from jax.lax import cond
from .jifty_convolution_operators import linpatch_convolve
from .data import Domain

try:
    from .adg.nifty_convolve import get_convolve
    adg_import = True
except ImportError:
    adg_import = False


def to_r_phi(cc):
    """
    Transforms form ra-dec (sky) coordinates to r-phi coordinates.
    """
    # FIXME: This assumes that ra is the x-coordinate and dec the y-coordinate
    # and furthermore assumes that the psfs are given in vertical distances
    # (off axis angle). Ensure that this is the correct orientation!
    r = jnp.sqrt(cc[..., 0]**2 + cc[..., 1]**2)
    phi = jnp.angle(cc[..., 0] + 1.j*cc[..., 1]) - jnp.pi/2.
    return jnp.stack((r, phi), axis = -1)


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
        int_coords = jnp.stack((dra, ddec), axis = -1)
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

        c_p = ((np.arange(ss) - ss/2)*dd for ss, dd in zip(shp, dist))
        centers = (np.array([(i*ss + ss/2)*dd for i in range(npatch)]) for
                   ss, dd in zip(patch_shp, dist))

        c_p = np.meshgrid(*c_p, indexing='ij')
        d_ra = c_p[0]
        d_dec = c_p[1]
        # Using 'xy' here instead of 'ij' ensures correct ordering as requested by
        # OAnew.
        centers = np.meshgrid(*centers, indexing='xy')
        c_ra = centers[0].flatten()
        c_dec = centers[1].flatten()

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

def _bilinear_weights(domain):
    """
    weights for the OverlapAdd Interpolation
    """
    psize = domain.shape[0]
    if psize/2 != int(psize/2):
        raise ValueError("this should happen")
    a = np.linspace(0, 1, int(psize/2), dtype="float64")
    b = np.concatenate([a, np.flip(a)])
    c = np.outer(b, b)
    return ift.Field.from_raw(domain, c)

def _get_weights(domain):
    """
    distribution the weights to the patch-space. Part of vectorization
    """
    weights = _bilinear_weights(domain[1])
    explode = ift.ContractionOperator(domain, spaces=0).adjoint
    return explode(weights)


class MarginZeroPadder(ift.LinearOperator):
    """
    ZeroPadder, adding zeros at the borders. This is different
    from other zero padding as soon as there are nor periodic
    boundary conditions.

    Parameters:
    ----------
    domain: NIFTy.RGSpace
    margin: int
    space: int

    return: operator
    """

    def __init__(self, domain, margin, space=0):
        self._domain = ift.makeDomain(domain)
        if not margin >= 1:
            raise ValueError("margin must be positive")
        self._margin = margin
        self._space = ift.utilities.infer_space(self.domain, space)
        dom = self._domain[self._space]
        old_shape = dom.shape
        new_shape = [k + 2 * margin for k in old_shape]
        self._target = list(self._domain)
        self._target[self._space] = ift.RGSpace(new_shape, dom.distances, dom.harmonic)
        self._target = ift.makeDomain(self._target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val
        curshp = list(self._dom(mode).shape)
        tgtshp = self._tgt(mode).shape
        for d in self._target.axes[self._space]:
            if v.shape[d] == tgtshp[d]:
                continue
            idx = (slice(None),) * d

            if mode == self.TIMES:
                shp = list(v.shape)
                shp[d] = tgtshp[d]
                xnew = np.zeros(shp, dtype=v.dtype)
                xnew[idx + (slice(self._margin, -self._margin),)] = v
            else:  # ADJOINT_TIMES
                xnew = v[idx + (slice(self._margin, -self._margin),)]
            curshp[d] = xnew.shape[d]
            v = xnew
        return ift.Field(self._tgt(mode), v)

class OAnew(ift.LinearOperator):
    """Operator for approximative inhomogeneous convolution.

    By OverlapAdd convolution with different kernels and bilinear
    interpolation of the result. In the case of one patch this simplifies
    to a regular Fourier domain convolution.

    Parameters:
    -----------
    domain: DomainTuple.
        Domain of the Operator
    kernel_arr: np.array
        Array containing the different kernels for the inhomogeneos convolution
    n: int
        Number of patches
    margin: int
        Size of the margin. Number of pixels on one boarder.

    HINT:
    The Operator checks if the kernel is zero in the regions not being used.
    If the initialization fails it can either be forced or cut by value.
    cut_force: sets all unused areas to zero,
    cut_by_value: sets everything below the threshold to zero.
    """

    def __init__(self, domain, kernel_arr, n, margin, want_cut):
        """Initialize the Overlap Add Operator."""
        self._domain = ift.makeDomain(domain)
        self._n = n
        self._op, self._cut = self._build_op(domain, kernel_arr, n, margin,
                                             want_cut)
        self._target = self._op.target
        self._capability = self.TIMES | self.ADJOINT_TIMES

    @staticmethod
    def _sqrt_n(n):
        sqrt = int(np.sqrt(n))
        if sqrt != np.sqrt(n):
            raise ValueError("""Operation is only defined on
                             square number of patches.""")
        return sqrt

    def apply(self, x, mode):
        """Apply the Operator."""
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = self._op(x)
        else:
            res = self._op.adjoint(x)
        return res

    @classmethod
    def _build_op(self, domain, kernel_arr, n, margin, want_cut):
        domain = ift.makeDomain(domain)
        oa = OverlapAdd(domain[0], n, 0)
        weights = ift.makeOp(_get_weights(oa.target))
        zp = MarginZeroPadder(oa.target, margin, space=1)
        padded = zp @ weights @ oa
        cutter = ift.FieldZeroPadder(
            oa.target, kernel_arr.shape[1:], space=1, central=True
        ).adjoint
        zp2 = ift.FieldZeroPadder(
            oa.target, padded.target.shape[1:], space=1, central=True
        )

        kernel_b = ift.Field.from_raw(cutter.domain, kernel_arr)
        if not self._check_kernel(domain, kernel_arr, n, margin):
            raise ValueError("""_check_kernel detected nonzero entries.
                             Use .cut_force, .cut_by_value!""")

        kernel = cutter(kernel_b)
        spread = ift.ContractionOperator(kernel.domain, spaces=1).adjoint
        norm = kernel.integrate(spaces=1)**-1
        norm = ift.makeOp(spread(norm))
        kernel = norm(kernel)
        kernel = zp2(kernel)

        convolved = convolve_field_operator(kernel, padded, space=1)
        pad_space = ift.RGSpace(
            [domain.shape[0] + 2 * margin, domain.shape[1] + 2 * margin],
            distances=domain[0].distances,
        )
        oa_back = OverlapAdd(pad_space, n, margin)
        extra_margin = (oa_back.domain.shape[0] - domain.shape[0])//2
        cut_pbc_margin = MarginZeroPadder(domain[0], extra_margin,
                                          space=0).adjoint

        if want_cut:
            interpolation_margin = (domain.shape[0]//self._sqrt_n(n))*2
            tgt_spc_shp = np.array(
                            [i-2*interpolation_margin for i in domain[0].shape])
            target_space = ift.RGSpace(tgt_spc_shp,
                                       distances=domain[0].distances)
            cut_interpolation_margin = (
                MarginZeroPadder(target_space, interpolation_margin).adjoint)
        else:
            cut_interpolation_margin = ift.ScalingOperator(
                                                    cut_pbc_margin.target, 1.)
        res = oa_back.adjoint @ convolved
        res = cut_interpolation_margin @ cut_pbc_margin @ res
        return res, cut_interpolation_margin

    @classmethod
    def cut_by_value(self, domain, kernel_list, n, margin, thrsh, want_cut):
        """Set the kernel zero for all values smaller than the threshold."""
        psfs = []
        for arr in kernel_list:
            arr[arr < thrsh] = 0
            psfs.append(arr)
        psfs = np.array(psfs, dtype="float64")
        if not self._check_kernel(domain, psfs, n, margin):
            raise ValueError("""_check_kernel detected nonzero entries.""")
        return OAnew(domain, psfs, n, margin, want_cut)

    @classmethod
    def cut_force(self, domain, kernel_list, n, margin, want_cut):
        """Set the kernel to zero where it is not used."""
        psfs = []
        nondef = self._psf_cut_area(domain, kernel_list, n, margin)
        for arr in kernel_list:
            arr[nondef == 0] = 0
            psfs.append(arr)
        psfs = np.array(psfs, dtype="float64")
        if not self._check_kernel(domain, psfs, n, margin):
            raise ValueError("""_check_kernel detected nonzero entries in areas
                            which should have been cut away.""")
        return OAnew(domain, psfs, n, margin, want_cut)

    @classmethod
    def _psf_cut_area(self, domain, kernel_list, n, margin):
        """Return the cut_area for psf.

        The returned is one where the psf gets cut.
        """
        psf_patch_shape = np.ones(np.array(domain.shape)//self._sqrt_n(n) * 2)
        psf_domain_shape = kernel_list[0].shape
        for d in [0, 1]:
            shp = list(psf_patch_shape.shape)
            idx = (slice(None),) * d
            shp[d] = psf_domain_shape[d]

            xnew = np.zeros(shp)
            Nyquist = psf_patch_shape.shape[d]//2
            i1 = idx + (slice(0, Nyquist+1),)
            xnew[i1] = psf_patch_shape[i1]
            i1 = idx + (slice(None, -(Nyquist+1), -1),)
            xnew[i1] = psf_patch_shape[i1]
            psf_patch_shape = xnew
        return psf_patch_shape

    @classmethod
    def _check_kernel(self, domain, kernel_list, n, margin):
        """Check if the kernel is appropriate for this method.

        For kernels being too large, this method is not suitable.
        """
        nondef = self._psf_cut_area(domain, kernel_list, n, margin)
        plist = []
        for p in kernel_list:
            arr = p
            nondef_arr = arr[nondef == 0]
            plist.append(nondef_arr)
        plist = np.array(plist, dtype="float64")
        return np.all(plist == 0)

def psf_lin_int_operator(domain, npatch, psf_infos, margfrac=0.1, 
                         want_cut=False, jaxop=True):
    """
    Psf convolution operator using bilinear interpolation of stationary patches.

    psf_infos : list of psf_infos(dict) generated by eROSITA PSF

    #FIXME could add a list of PSF Infos or an NDArray of in the psfinfo already
    """
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

        c_p = ((np.arange(ss) - ss/2)*dd for ss, dd in zip(shp, dist))
        centers = (np.array([(i*ss + ss/2)*dd for i in range(npatch)]) for
                   ss, dd in zip(patch_shp, dist))

        c_p = np.meshgrid(*c_p, indexing='ij')
        d_ra = c_p[0]
        d_dec = c_p[1]
        # Using 'xy' here instead of 'ij' ensures correct ordering as requested by
        # OAnew.
        centers = np.meshgrid(*centers, indexing='xy')
        c_ra = centers[0].flatten()
        c_dec = centers[1].flatten()

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
    margin = max(good_fft_size(int(np.ceil(margfrac*ss))) for ss in shp)
    if jaxop:
        # TODO Want cut?
        n_patches_per_axis = int(np.sqrt(len(patch_psfs)))

        def op(x):
            return linpatch_convolve(x, domain, all_patches,
                                     n_patches_per_axis, margin)
    else:
        # FIXME OAnew for Energies probably only on xubik / but also here for testing?
        op = OAnew(domain, patch_psfs, len(patch_psfs), margin, want_cut)
    return op

