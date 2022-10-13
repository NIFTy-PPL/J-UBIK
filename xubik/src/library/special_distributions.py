# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import nifty8 as ift
#import nifty8.re as jft
from scipy.stats import invgamma, norm
from nifty8.library.special_distributions import _InterpolationOperator


class InverseGammaOperator(ift.Operator):
    """Transform a standard normal into an inverse gamma distribution with inferred parameter q.

    The pdf of the inverse gamma distribution is defined as follows:

    .. math::
        \\frac{q^\\alpha}{\\Gamma(\\alpha)}x^{-\\alpha -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^{(-\\alpha -1)}`.
    The mean of the pdf is at :math:`q / (\\alpha - 1)` if :math:`\\alpha > 1`.
    The mode is :math:`q / (\\alpha + 1)`.

    The operator can be initialized by setting either alpha and q or mode and mean.
    In accordance to the statements above the mean must be greater
    than the mode. Otherwise would get alpha < 0 and so no mean would be defined.

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto an inverse gamma distribution.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    mean_q : float
       The mean of the inferred parameter q
    std_q : float
        The standard deviation  of the inferred parameter q
    alpha : float
        The alpha-parameter of the inverse-gamma distribution.
    delta : float
        Distance between sampling points for linear interpolation.
    """
    def __init__(self, domain, mean_q, std_q, alpha, delta=1e-2):
        self._target = ift.DomainTuple.make(domain)
        if isinstance(mean_q, ift.Field):
            q_domain = mean_q.domain
        else:
            q_domain = ift.DomainTuple.scalar_domain()
        domain_dict = {'points_q': q_domain, 'points': domain}
        self._domain = ift.MultiDomain.make(domain_dict)
        self._mean_q = mean_q
        self._alpha = float(alpha)
        self._delta = float(delta)
        op = _InterpolationOperator(self.target, lambda x: invgamma.ppf(norm._cdf(x), float(self._alpha)),
                                    -8.2, 8.2, self._delta, lambda x: x.ptw("log"), lambda x: x.ptw("exp"))
        op = op.ducktape('points')

        q_op = ift.LognormalTransform(self._mean_q, std_q, list(domain_dict.keys())[0], N_copies=0)
        self.q_op = q_op.exp()
        expanded_q_op = ift.ContractionOperator(domain_dict['points'], None).adjoint(self.q_op)
        self._op = expanded_q_op * op
        #TODO:Jax-Version

    def apply(self, x):
        return self._op(x)

    @property
    def alpha(self):
        """float : The value of the alpha-parameter of the inverse-gamma distribution"""
        return self._alpha

    @property
    def mean_q(self):
         """float : The value of the mean of the q-parameters of the inverse-gamma distribution"""
         return self._mean_q
    def q(self):
        """operator : Operator to the value of the q-parameter"""
        return self.q_op


class InverseGammaOperator_alpha(ift.Operator):
    """Transform a standard normal into an inverse gamma distribution with inferred parameter q.

    The pdf of the inverse gamma distribution is defined as follows:

    .. math::
        \\frac{q^\\alpha}{\\Gamma(\\alpha)}x^{-\\alpha -1}
        \\exp \\left(-\\frac{q}{x}\\right)

    That means that for large x the pdf falls off like :math:`x^{(-\\alpha -1)}`.
    The mean of the pdf is at :math:`q / (\\alpha - 1)` if :math:`\\alpha > 1`.
    The mode is :math:`q / (\\alpha + 1)`.

    The operator can be initialized by setting either alpha and q or mode and mean.
    In accordance to the statements above the mean must be greater
    than the mode. Otherwise would get alpha < 0 and so no mean would be defined.

    This transformation is implemented as a linear interpolation which maps a
    Gaussian onto an inverse gamma distribution.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain on which the field shall be defined. This is at the same
        time the domain and the target of the operator.
    mean_q : float
       The mean of the inferred parameter q
    std_q : float
        The standard deviation  of the inferred parameter q
    alpha : float
        The alpha-parameter of the inverse-gamma distribution.
    delta : float
        Distance between sampling points for linear interpolation.
    """
    def __init__(self, domain, mean_q, std_q, alpha, delta=1e-2):
        self._target = ift.DomainTuple.make(domain)
        if isinstance(mean_q, ift.Field):
            q_domain = mean_d.domain
        else:
            q_domain = ift.DomainTuple.scalar_domain()
        domain_dict = {'points_q': q_domain, 'points': domain}
        self._domain = ift.MultiDomain.make(domain_dict)
        self._mean_q = mean_q
        self._alpha = float(alpha)
        self._delta = float(delta)
        op = _InterpolationOperator(self.target, lambda x: invgamma.ppf(norm._cdf(x), float(self._alpha)),
                                    -8.2, 8.2, self._delta, lambda x: x.ptw("log"), lambda x: x.ptw("exp"))
        op = op.ducktape('points')

        q_op = ift.LognormalTransform(self._mean_q, std_q, list(domain_dict.keys())[0], N_copies=0)
        self.q_op = q_op.exp()
        expanded_q_op = ift.ContractionOperator(domain_dict['points'], None).adjoint(self.q_op)
        self._op = expanded_q_op * op
        #TODO:Jax-Version

    def apply(self, x):
        return self._op(x)

    @property
    def alpha(self):
        """float : The value of the alpha-parameter of the inverse-gamma distribution"""
        return self._alpha

    @property
    def mean_q(self):
         """float : The value of the mean of the q-parameters of the inverse-gamma distribution"""
         return self._mean_q

    def q(self):
        """operator : Operator to the value of the q-parameter"""
        return self.q_op
