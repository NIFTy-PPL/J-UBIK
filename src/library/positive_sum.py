import numpy as np
import scipy

import nifty8 as ift


"""This File needs to be transfered to nifty.re"""

class PositiveSumPriorOperator(ift.LinearOperator):
    """
    Operator performing a coordinate transformation, requiring MultiToTuple
    and PositiveSumTrafo. The operator takes the input, here a nifty8.MultiField, mixes
    it using a coordinate tranformation and spits out a nifty8.MultiField
    again.
    """
    def __init__(self, domain, target=None):
        """
        Creates the Operator.

        Paramters
        ---------
        domain: nifty8.MultiDomain
        target: nifty8.MultiDomain
            Default: target == domain

        """
        self._domain = domain
        if not isinstance(self._domain, ift.MultiDomain):
            raise TypeError("domain must be a MultiDomain")
        if target == None:
            self._target = self._domain
        else:
            self._target = target
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._multi = MultiToTuple(self._domain)
        self._trafo = PositiveSumTrafo(self._multi.target)

    def apply(self, x, mode):
        self._check_input(x, mode)
        op = self._multi.adjoint @ self._trafo @ self._multi
        if mode == self.TIMES:
            res = op(x)
        else:
            res = op.adjoint(x)
        return res


class MultiToTuple(ift.LinearOperator):
    """
    Puts several Fields of a Multifield of the same domains, into a DomainTuple
    along a UnstructuredDomain. It's adjoint reverses the action.
    """

    def __init__(self, domain):
        """
        Creates the Operator.

        Paramters
        ---------
        domain: nifty8.MultiDomain
        """
        self._domain = domain
        if not isinstance(self._domain, ift.MultiDomain):
            raise TypeError("domain has to be a ift.MultiDomain")
        self._first_dom = domain[domain.keys()[0]][0]
        for key in self._domain.keys():
            if not self._first_dom == domain[key][0]:
                raise TypeError("All sub domains must be equal ")
        n_doms = ift.UnstructuredDomain(len(domain.keys()))
        self._target = ift.makeDomain((n_doms, self._first_dom))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            lst = []
            for key in x.keys():
                lst.append(x[key].val)
            x = np.array(lst)
            res = ift.Field.from_raw(self._target, x)
        else:
            dct = {}
            ii = 0
            for key in self._domain.keys():
                tmp_field = ift.Field.from_raw(self._first_dom, x.val[ii, :, :])
                dct.update({key: tmp_field})
                ii += 1
            res = ift.MultiField.from_dict(dct)
        return res


class PositiveSumTrafo(ift.EndomorphicOperator):
    """
    This Operator performs a coordinate transformation into a coordinate
    system, in which the Oth component is the sum of all components of
    the former basis. Can be used as a replacement of softmax.
    """
    def __init__(self, domain):
        """
        Creates the Operator.

        Parameters
        ----------
        domain: nifty8.domain
        """
        self._domain = ift.makeDomain(domain)
        self._n = self.domain.shape[0]
        self._build_mat()
        self._capability = self.TIMES | self.ADJOINT_TIMES
        lamb, s = self._build_mat()
        self._lamb = lamb
        if not np.isclose(lamb[0], 0):
            raise ValueError(
                "Transformation does not work, check eigenvalues self._lamb"
            )
        self._s = s
        if s[0, 0] < 0:
            s[:, 0] = -1 * s[:, 0]
        self._s_inv = scipy.linalg.inv(s)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            y = np.einsum("ij, jmn->imn", self._s, x)
        else:
            y = np.einsum("ij, jmn-> imn", self._s_inv, x)
        return ift.Field.from_raw(self._tgt(mode), y)

    def _build_mat(self):
        l = self._n
        one = np.zeros([l] * 2)
        np.fill_diagonal(one, 1)

        norm_d = np.ones([l] * 2) / l
        proj = one - norm_d
        eigv, s = np.linalg.eigh(proj)
        return eigv, s


def get_distributions_for_positive_sum_prior(domain, number):
    """
    Builds Priors for the PositiveSumTrafo Operator. Here the 0th Component is
    supposed to be sum of all others. Since we want the sum to be positive,
    but some of the summands may be negative. Therefore the 0th component is a
    priori log-normal distributed.

    Parameters
    ----------
    domain: nifty8.domain
        Domain of each component
    number: int
        number of components

    Returns
    -------
    nifty8.OpChain
        Part of the generative model.
    """
    for i in range(number):
        field_adapter = ift.FieldAdapter(domain, f"amp_{i}")
        tmp_operator = field_adapter.adjoint @ field_adapter
        if i == 0:
            operator = tmp_operator.exp()
        else:
            operator = operator + tmp_operator
    return operator


def makePositiveSumPrior(domain, number):
    """
    Convenience function to combine PositiveSumPriorOperator and
    get_distributions_for_prior.

    Paramters
    ---------
    domain: nifty8.domain
        Domain of one component, which will be mixed.
    number: int
        Number of components

    Returns
    -------
    nifty8.OpChain
    """
    distributions = get_distributions_for_positive_sum_prior(domain, number)
    positive_sum = PositiveSumPriorOperator(distributions.target)
    op = positive_sum @ distributions
    return op
