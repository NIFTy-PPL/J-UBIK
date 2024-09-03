import numpy as np
import nifty8 as ift

from .zero_padder import MarginZeroPadder
from ..library.utils import convolve_field_operator


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


class OverlapAdd(ift.LinearOperator):
    """Slicing operator for linear patched convolution.

    The operator slices a 2D array into N patches with dx offset and
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
    Comparable to PatchingOperator + Overlapp + Margins
    """

    def __init__(self, domain, n_patch, pbc_margin):
        """Inatialize the slicing operator."""
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
        """Apply sclicing."""
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
            taped_s += taped[self.dx // 2: -self.dx // 2, self.dy // 2: -self.dy // 2]
            res = ift.Field.from_raw(self._domain, taped_s)
        return res