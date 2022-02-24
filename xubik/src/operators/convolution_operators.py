import nifty8 as ift
import numpy as np

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
        shape = domain.shape
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

    def coord_center(self):
        xc = np.arange(self.dx // 2, self.dx * self.sqrt_n_patch, self.dx)
        yc = np.arange(self.dy // 2, self.dy * self.sqrt_n_patch, self.dy)
        co = np.array(np.meshgrid(xc, yc)).reshape(2,-1)
        return co
