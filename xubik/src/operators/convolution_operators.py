import numpy as np

import nifty8 as ift

from .zero_padder import MarginZeroPadder
from .bilinear_interpolation import get_weights
from ..library.utils import convolve_field_operator


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

    # TODO add checks and test
    # FIXME Restructure
    # FIXME omit loops

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
        dx, dy = self.dx, self.dy
        if mode == self.TIMES:
            xplus = np.zeros([self._domain.shape[0] + self.dx] * 2)
            # TODO think about doing without this dx for the bordes and odd number of patches?
            xplus[
                self.dx // 2 : xplus.shape[0] - self.dx // 2,
                self.dy // 2 : xplus.shape[1] - self.dy // 2,
            ] = x.val
            listing = []
            for l in range(self.sqrt_n_patch):
                y_i = l * dy
                y_f = y_i + 2 * dy + 2 * self.dr
                for k in range(self.sqrt_n_patch):
                    x_i = k * dx
                    x_f = x_i + 2 * dx + 2 * self.dr
                    tmp = xplus[x_i:x_f, y_i:y_f]
                    listing.append(tmp)
            array = np.array(listing)
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
                    taped[x_i:x_f, y_i:y_f] += x.val[i]
                    i += 1
            taped_s = np.zeros(self.domain.shape)
            taped_s += taped[self.dx // 2 : -self.dx // 2, self.dy // 2 : -self.dy // 2]
            res = ift.Field.from_raw(self._domain, taped_s)
        return res


def OverlapAddConvolver(domain, kernels_arr, n, margin):
    """
    Performing a approximation to an inhomogeneous convolution,
    by OverlapAdd convolution with different kernels and bilinear
    interpolation of the result. In the case of one patch this simplifies
    to a regular Fourier domain convolution.

    Parameters:
    -----------
    domain: DomainTuple.
        Domain of the Operator
    kernels_arr: np.array
        Array containing the different kernels for the inhomogeneos convolution
    n: int
        Number of patches
    margin: int
        Size of the margin. Number of pixels on one boarder.

    """
    oa = OverlapAdd(domain[0], n, 0)
    weights = ift.makeOp(get_weights(oa.target))
    zp = MarginZeroPadder(oa.target, margin, space=1)
    padded = zp @ weights @ oa
    cutter = ift.FieldZeroPadder(
        padded.target, kernels_arr.shape[1:], space=1, central=True
    ).adjoint
    kernels_b = ift.Field.from_raw(cutter.domain, kernels_arr)
    kernels = cutter(kernels_b)
    convolved = convolve_field_operator(kernels, padded, space=1)
    pad_space = ift.RGSpace(
        [domain.shape[0] + 2 * margin, domain.shape[1] + 2 * margin],
        distances=domain[0].distances,
    )
    oa_back = OverlapAdd(pad_space, n, margin)
    res = oa_back.adjoint @ convolved
    return res
