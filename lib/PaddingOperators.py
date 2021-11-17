import nifty8 as ift
import numpy as np


class MarginZeroPadder(ift.LinearOperator):
    """ZeroPadder, adding zeros at the borders"""

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
