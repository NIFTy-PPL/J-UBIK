import nifty8 as ift
import matplotlib.pylab as plt
import numpy as np

ps_spectrum_loglinear_slope = {
    # on log10/log10 scale
    'mean': -2.43,
    'sigma': 0.25,
    'key': 'ps_spectrum_slope_xi'
}

ps_spectrum_cfm = {
    'offset_mean': 0.,
    'offset_std_mean': 1e-3,
    'offset_std_std': 1e-16,
    'prefix': 'ps_spectrum_fluctuations_'
}

ps_spectrum_fluctuations = {
    'fluctuations_mean': 0.05,
    'fluctuations_stddev': 0.05,
    'flexibility_mean': 0.02,
    'flexibility_stddev': 0.02,
    'asperity_mean': 0.01,
    'asperity_stddev': 0.01,
    'loglogavgslope_mean': -1.5,
    'loglogavgslope_stddev': 0.25,
    'prefix': ''
}
ps_brightness = {'alpha': 1.5, 'q': 1e-7}

e_space = ift.RGSpace([9])
space = ift.RGSpace([32,32])
x_npix = space.shape[0]

tot_delta_log10_E = e_space.total_volume
delta_log10_E = e_space.distances[0]
unit_slope_ps = np.arange(0., tot_delta_log10_E, delta_log10_E)
unit_slope_ps -= tot_delta_log10_E / 2
unit_slope_ps = ift.makeField(e_space, unit_slope_ps)

point_bright = ift.InverseGammaOperator(space, **ps_brightness)

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

def transform_loglog_slope_pars(slope_pars):
        """The slope parameters given in the config are
        for slopes in log10/log10 space.
        However, since the energy bins are log10-spaced
        and the signal is modeled in ln-space, the parameters
        have to be transformed prior to their use."""
        res = slope_pars.copy()
        res['mean'] = (res['mean'] + 1) * np.log(10)
        res['sigma'] *= np.log(10)

        return res

ps_spectra_loglinear_slopes = ift.NormalTransform(
    N_copies=x_npix,
    **transform_loglog_slope_pars(ps_spectrum_loglinear_slope))
slopeOp = ReverseOuterProduct(ps_spectra_loglinear_slopes.target,
                              unit_slope_ps)
ps_spectra_loglinear_part = ps_spectra_loglinear_slopes


sp = ift.UnstructuredDomain(1000)
op = ift.NormalTransform(N_copies=100000, mean=4, sigma=3, key='')
f = ift.from_random(op.domain)
g = op(f)
