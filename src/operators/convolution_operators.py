import numpy as np

import nifty8 as ift

from .zero_padder import MarginZeroPadder
from ..library.utils import convolve_field_operator


def _bilinear_weights(domain):
    psize = domain.shape[0]
    if psize/2 != int(psize/2):
        raise ValueError("this should happen")
    a = np.linspace(0, 1, int(psize/2), dtype="float64")
    b = np.concatenate([a, np.flip(a)])
    c = np.outer(b,b)
    return ift.Field.from_raw(domain, c)

def _get_weights(domain):
    weights = _bilinear_weights(domain[1])
    explode = ift.ContractionOperator(domain, spaces=0).adjoint
    return explode(weights)

class OverlapAdd(ift.LinearOperator):
    """Slices a 2D array into N patches with dx offset and
    2*dx+2*dr side length and arranges them in a new space (unstructured).

    Parameters:
    ----------
    domain: DomainTuple
        Domain of the Operator.
    n_patch: int
        number of patches after the slicing operation
    pbc_margin:
        additional margin in order to break

    Notes:
    ------
    Comparable to PatchingOperator + Overlapp + Margins"""


    def __init__(self, domain, n_patch, pbc_margin):
        self._domain = ift.makeDomain(domain)
        self.sqrt_n_patch = int(np.sqrt(n_patch))
        self.dr = pbc_margin
        self.dx, self.dy = [
            int((domain.shape[0] - 2 * self.dr) / self.sqrt_n_patch)
        ] * 2
        small_space = ift.RGSpace(
            [int((domain.shape[0] - 2 * self.dr) / self.sqrt_n_patch) * 2 + 2 * self.dr]
            * 2,
            domain.distances,
        )
        patch_space = ift.UnstructuredDomain(n_patch)
        self._target = ift.makeDomain([patch_space, small_space])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        val = np.copy(x.val)
        dx, dy = self.dx, self.dy
        if mode == self.TIMES:
            xplus = np.zeros([self._domain.shape[0] + self.dx] * 2)
            # TODO think about doing without this dx for the bordes
            # and odd number of patches?
            xplus[
                self.dx // 2: xplus.shape[0] - self.dx // 2,
                self.dy // 2: xplus.shape[1] - self.dy // 2,
            ] = val
            listing = []
            for l in range(self.sqrt_n_patch):
                y_i = l * dy
                y_f = y_i + 2 * dy + 2 * self.dr
                for k in range(self.sqrt_n_patch):
                    x_i = k * dx
                    x_f = x_i + 2 * dx + 2 * self.dr
                    tmp = xplus[x_i:x_f, y_i:y_f]
                    listing.append(tmp)
            res = ift.Field.from_raw(self._target, np.array(listing))
        else:
            taped = np.zeros([self._domain.shape[0] + self.dx] * 2)
            i = 0
            for n in range(self.sqrt_n_patch):
                y_i = n * dy
                y_f = y_i + 2 * dy + 2 * self.dr
                for m in range(self.sqrt_n_patch):
                    x_i = m * dx
                    x_f = x_i + 2 * dx + 2 * self.dr
                    taped[x_i:x_f, y_i:y_f] += val[i]
                    i += 1
            taped_s = np.zeros(self.domain.shape)
            taped_s += taped[self.dx // 2 : -self.dx // 2, self.dy // 2 : -self.dy // 2]
            res = ift.Field.from_raw(self._domain, taped_s)
        return res


class OAConvolver(ift.LinearOperator):
    """
    Performing a approximation to an inhomogeneous convolution,
    by OverlapAdd convolution with different kernels and bilinear
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
    def __init__(self, domain, kernel_arr, n, margin):
        self._domain = ift.makeDomain(domain)
        self._n = n
        self._op = self._build_op(domain, kernel_arr, n, margin)
        self._target = self._op.target
        self._capability = self.TIMES | self.ADJOINT_TIMES

    @staticmethod
    def _sqrt_n(n):
        sqrt = int(np.sqrt(n))
        if sqrt != np.sqrt(n):
            raise ValueError("Operation is only defined on square number of patches.")
        return sqrt

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = self._op(x)
        else:
            res = self._op.adjoint(x)
        return res

    @classmethod
    def _build_op(self, domain, kernel_arr, n, margin):
        domain = ift.makeDomain(domain)
        pad_margin = MarginZeroPadder(domain[0], margin, space=0)
        oa = OverlapAdd(pad_margin.target[0], n, margin)
        patched = oa @ pad_margin

        cutter = ift.FieldZeroPadder(
            patched.target, kernel_arr.shape[1:], space=1, central=True
        ).adjoint
        # TODO Kernels as IFT Field
        kernels_b = ift.Field.from_raw(cutter.domain, kernel_arr)

        if not self._check_kernel(domain, kernel_arr, n, margin):
            raise ValueError("_check_kernel detected nonzero entries. Use .cut_force, .cut_by_value!")

        kernels = cutter(kernels_b)
        spread = ift.ContractionOperator(kernels.domain, spaces=1).adjoint
        norm = kernels.integrate(spaces=1)**-1
        norm = ift.makeOp(spread(norm))
        kernels = norm(kernels)

        convolved = convolve_field_operator(kernels, patched, space=1)
        n_space = ift.UnstructuredDomain(n)
        p_space = ift.RGSpace(np.array(domain.shape)//self._sqrt_n(n) * 2, distances=domain[0].distances)
        pad_space = ift.makeDomain([n_space, p_space])

        cut_conv_margin = MarginZeroPadder(pad_space, margin, space=1).adjoint
        convolved = cut_conv_margin(convolved)
        weights = ift.makeOp(_get_weights(convolved.target))
        weighted = weights @ convolved
        oa_back = OverlapAdd(domain[0], n, 0)

        tgt_spc_shp = np.array([i-2*margin for i in domain[0].shape])
        target_space = ift.RGSpace(tgt_spc_shp, distances=domain[0].distances)
        cut_interpolation_margin = MarginZeroPadder(target_space, margin).adjoint

        return cut_interpolation_margin @ oa_back.adjoint @ weighted

    @classmethod
    def cut_by_value(self, domain, kernel_list, n, margin, thrsh):
        """
        Sets the kernel zero for all values smaller than the threshold and initializes the operator.
        """
        psfs = []
        for p in kernel_list:
            arr = p#.val_rw()
            arr[arr < thrsh] = 0
            psfs.append(arr)
        psfs = np.array(psfs, dtype="float64")
        if not self._check_kernel(domain, psfs, n, margin):
            raise ValueError("_check_kernel detected nonzero entries. Lower threshold or check kernel!")
        return OAConvolver(domain, psfs, n, margin)

    @classmethod
    def cut_force(self, domain, kernel_list, n, margin):
        """
        Sets the kernel to zero where it is not used and initializes the operator.
        """
        psfs = []
        nondef = self._psf_cut_area(domain, kernel_list, n, margin)
        for p in kernel_list:
            arr = p#.val_rw()
            arr[nondef == 0] = 0
            psfs.append(arr)
        psfs = np.array(psfs, dtype="float64")
        if not self._check_kernel(domain, psfs, n, margin):
            raise ValueError("_check_kernel detected nonzero entries in areas which should have been cut away. Please double check!")
        return OAConvolver(domain, psfs, n, margin)

    @classmethod
    def _psf_cut_area(self, domain, kernel_list, n, margin):
        """
        Returns an array with size of the full kernel.
        It is one where the psf gets cut out.
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
        """
        checks if the kernel is appropriate for this method.
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

# TODO Think about what is the right thing and fix the stuff being wrong in the lower op

# class OverlapAddConvolver(ift.LinearOperator):
#     """
#     Performing a approximation to an inhomogeneous convolution,
#     by OverlapAdd convolution with different kernels and bilinear
#     interpolation of the result. In the case of one patch this simplifies
#     to a regular Fourier domain convolution.

#     Parameters:
#     -----------
#     domain: DomainTuple.
#         Domain of the Operator
#     kernels_arr: np.array
#         Array containing the different kernels for the inhomogeneos convolution
#     n: int
#         Number of patches
#     margin: int
#         Size of the margin. Number of pixels on one boarder.

#     """
#     def __init__(self, domain, kernels_arr, n, margin):
#         self._domain = domain
#         self._op = self._build_op(domain, kernels_arr, n, margin)
#         self._target = self._op.target
#         self._capability = self.TIMES | self.ADJOINT_TIMES
#         # TODO check for area cut out => only 0
#         # TODO normlize

#     def apply(self, x, mode):
#         self._check_input(x, mode)
#         if mode == self.TIMES:
#             res = self._op(x)
#         else:
#             res = self._op.adjoint(x)
#         return res

#     @staticmethod
#     def _build_op(domain, kernels_arr, n, margin):
#         domain = ift.makeDomain(domain)
#         oa = OverlapAdd(domain[0], n, 0)
#         weights = ift.makeOp(get_weights(oa.target))
#         zp = MarginZeroPadder(oa.target, margin, space=1)
#         padded = zp @ weights @ oa
#         cutter = ift.FieldZeroPadder(
#             padded.target, kernels_arr.shape[1:], space=1, central=True
#         ).adjoint
#         kernels_b = ift.Field.from_raw(cutter.domain, kernels_arr)
#         kernels = cutter(kernels_b)
#         convolved = convolve_field_operator(kernels, padded, space=1)
#         pad_space = ift.RGSpace(
#             [domain.shape[0] + 2 * margin, domain.shape[1] + 2 * margin],
#             distances=domain[0].distances,
#         )
#         oa_back = OverlapAdd(pad_space, n, margin)
#         cut = MarginZeroPadder(domain[0], ((oa_back.domain.shape[0] - domain.shape[0])//2), space=0).adjoint
#         res = cut @ oa_back.adjoint @ convolved
#         return res

class OAnew(ift.LinearOperator):
    """
    Performing a approximation to an inhomogeneous convolution,
    by OverlapAdd convolution with different kernels and bilinear
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
            raise ValueError("Operation is only defined on square number of patches.")
        return sqrt

    def apply(self, x, mode):
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
            raise ValueError("_check_kernel detected nonzero entries. Use .cut_force, .cut_by_value!")

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
        cut_pbc_margin = MarginZeroPadder(domain[0], 
            ((oa_back.domain.shape[0] - domain.shape[0])//2), space=0).adjoint
        
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
    def cut_by_value(self, domain, kernel_list, n, margin, thrsh):
        """
        Sets the kernel zero for all values smaller than the threshold and initializes the operator.
        """
        psfs = []
        for p in kernel_list:
            arr = p#.val_rw()
            arr[arr < thrsh] = 0
            psfs.append(arr)
        psfs = np.array(psfs, dtype="float64")
        if not self._check_kernel(domain, psfs, n, margin):
            raise ValueError("_check_kernel detected nonzero entries. Lower threshold or check kernel!")
        return OAnew(domain, psfs, n, margin)

    @classmethod
    def cut_force(self, domain, kernel_list, n, margin):
        """
        Sets the kernel to zero where it is not used and initializes the operator.
        """
        psfs = []
        nondef = self._psf_cut_area(domain, kernel_list, n, margin)
        for p in kernel_list:
            arr = p#.val_rw()
            arr[nondef == 0] = 0
            psfs.append(arr)
        psfs = np.array(psfs, dtype="float64")
        if not self._check_kernel(domain, psfs, n, margin):
            raise ValueError("_check_kernel detected nonzero entries in areas which should have been cut away. Please double check!")
        return OAnew(domain, psfs, n, margin)

    @classmethod
    def _psf_cut_area(self, domain, kernel_list, n, margin):
        """
        Returns an array with size of the full kernel.
        It is one where the psf gets cut out.
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
        """
        checks if the kernel is appropriate for this method.
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
