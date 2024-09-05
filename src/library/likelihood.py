import nifty8 as ift
import nifty8.re as jft

from .data import load_masked_data_from_config
from .instruments.erosita.erosita_data import generate_erosita_data_from_config
from .. import build_erosita_response_from_config

def get_n_constrained_dof(likelihood: jft.Likelihood) -> int:
    """
    Extacts the number of constrained degrees of freedom (DOF)
    based on the likelihood.

    Parameters
    ----------
    likelihood : jft.Likelihood
        The likelihood object which contains information about
       the model and data.

    Returns
    -------
    int
        The number of constrained degrees of freedom, which is the
        minimum of the model degrees of freedom and the data
        degrees of freedom.
    """

    n_dof_data = jft.size(likelihood.left_sqrt_metric_tangents_shape)
    n_dof_model = jft.size(likelihood.domain)
    return min(n_dof_model, n_dof_data)


# TODO to nifty.re
class _IGLikelihood(ift.EnergyOperator):
    """
    Functional form of the Inverse-Gamma distribution. Can be used for
    Equal-likelihood-optimization.

    Notes:
    ------
        This implementation is only designed for a point source component over
        a single-frequency sky.
    """
    # TODO: Build MF-version

    def __init__(self, data, alpha, q):
        """
        Constructs an EnergyOperator specifially for InverseGamma Likelihoods.

        Parameters
        ----------
        data: nifty8.Field
        alpha: float
        q: float
        """
        self._domain = ift.makeDomain(data.domain)
        shift = ift.Adder(data) @ ift.ScalingOperator(self._domain, -1)
        dummy = ift.ScalingOperator(self._domain, 1)
        self._q = ift.ScalingOperator(self._domain, float(q))
        self._apw = ift.ScalingOperator(self._domain, float(alpha + 1.))
        op = self._q @ dummy.ptw('reciprocal') + self._apw @ dummy.ptw('log')
        self._op = (op @ shift.ptw('abs')).sum()

    def apply(self, x):
        self._check_input(x)
        res = self._op(x)
        if not x.want_metric:
            return res
        raise NotImplementedError

# TODO to nifty.re
def get_equal_lh_transition(sky, diffuse_sky, point_dict, transition_dict,
                            point_key = 'point_sources', stiffness = 1E6,
                            red_factor = 1E-3):
    """
    Performs a likelihood (i.E. input sky) invariant transition between the
    dofs of a diffuse component and point sources. Assumes `sky`to be composed
    as
        sky = diffuse_sky(xi_diffuse) + point_sky(xi_point)

    where `(..._)sky` are `nifty` Operators and `xi` are the standard dofs of
    the components. The operator `point_sky` is assumed to be a generative
    process for an Inverse-Gamma distribution matching the convention of
    `_IGLikelihood`.

    Parameters:
    -----------
        sky: nifty8.Operator
            Generative model for the sky consisting of a point source component
            and another additive component `diffuse_sky`.
        diffuse_sky: nifty8.Operator
            Generative model describing only the diffuse component.
        point_dict: dict of float
            Dictionary containing the Inverse-Gamma parameters `alpha` and `q`.
        transition_dict: dict
            Optimization parameters for the iteration controller of the
            transition optimization loop.
        point_key: str (default: 'point_sources')
            Key of the point source dofs in the MultiField of the joint
            reconstruction.
        stiffness: float (default: 1E6)
            Precision of the point source dof optimization after updating the
            diffuse components
        red_factor: float (default: 1E-3)
            Scaling for the convergence criterion regarding the second
            optimization for the point source dofs.
    """
    # TODO: replace second optimization with proper inverse transformation!
    def _transition(position):
        diffuse_pos = position.to_dict()
        diffuse_pos.pop(point_key)
        diffuse_pos = ift.MultiField.from_dict(diffuse_pos)

        my_sky = sky(position)

        lh = _IGLikelihood(my_sky, point_dict['alpha'], point_dict['q'])

        ic_mini = ift.AbsDeltaEnergyController(
                        deltaE = float(transition_dict['deltaE']),
                        iteration_limit = transition_dict['iteration_limit'],
                        convergence_level=transition_dict['convergence_level'],
                        name = transition_dict['name'])
        ham = ift.StandardHamiltonian(lh @ diffuse_sky)
        en, _ = ift.VL_BFGS(ic_mini)(ift.EnergyAdapter(diffuse_pos, ham))
        diffuse_pos = en.position

        new_pos = diffuse_pos.to_dict()
        new_pos['point_sources'] = position['point_sources']
        new_pos = ift.MultiField.from_dict(new_pos)

        icov = ift.ScalingOperator(my_sky.domain, stiffness)
        lh = ift.GaussianEnergy(data=my_sky, inverse_covariance=icov)
        en = ift.EnergyAdapter(new_pos, lh @ sky,
                               constants=list(diffuse_pos.keys()))
        ic_mini = ift.AbsDeltaEnergyController(
                        deltaE = red_factor * float(transition_dict['deltaE']),
                        iteration_limit = transition_dict['iteration_limit'],
                        convergence_level=transition_dict['convergence_level'],
                        name = transition_dict['name'])

        new_point_source_position = ift.VL_BFGS(ic_mini)(en)[0].position.to_dict()
        new_pos = new_pos.to_dict()
        new_pos['point_sources'] = new_point_source_position['point_sources']
        return ift.MultiField.from_dict(new_pos)

    _tr = (lambda samples: samples.average(_transition))
    return lambda iiter: None if iiter < transition_dict['start'] else _tr
