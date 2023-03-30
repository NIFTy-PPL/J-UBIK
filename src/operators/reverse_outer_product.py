import nifty8 as ift
import numpy as np

# TODO Could we rename this class? This sounds wrong to me
class ReverseOuterProduct(ift.LinearOperator):
    """Performs the point-wise outer product `field x inp_field`.

    Parameters
    ---------
    field: Field,
    domain: DomainTuple, the domain of the input field
    ---------
    """
    def __init__(self, domain, field):
        self._domain = domain
        self._field = field

        self._target = ift.DomainTuple.make(
            tuple(sub_d for sub_d in domain._dom + field.domain._dom))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return ift.Field(self._target,
                             np.multiply.outer(x.val, self._field.val))
        axes = len(self._field.shape)  # only valid for len(domain.shape) == 1
        return ift.Field(self._domain,
                         np.tensordot(x.val, self._field.val, axes))
