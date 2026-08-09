"""Weight transformations for resolve observations."""

import numpy as np
from nifty.cl.logger import logger

from ...parse.data.data_modify.weights import SystematicErrorBudget
from ..observation import Observation


def systematic_error_budget(obs: Observation, systematic: SystematicErrorBudget | None):
    """Modify the weights of the observation. The weights get an added standard
    deviation of `systematic.percentage` of the amplitude of the
    visibilities.

    Note
    ----
    weight = 1 / (sigma**2 + (systematic.percentage * |A|)**2 )


    Parameters
    ----------
    obs: Observation
        The observation to modifiy
    systematic: SystematicErrorBudget
        The parameters of the weight modify class, holding the percentage of
        the amplitude fraction.
    """

    if systematic is None:
        return obs

    logger.info(
        "Applied systematic error budget by "
        f"{systematic.percentage * 100} percent (sigma^2+(perc*|A|)^2)."
    )

    weight_old = obs.weight.asnumpy()
    perc = systematic.percentage

    # Entries with weight == 0 are flagged and stay flagged. They are excluded
    # from the division to avoid inf/nan showing up in the new weights.
    valid = weight_old > 0.0

    # sigma**2 = 1/weight
    sigma_squared = np.divide(
        1.0, weight_old, out=np.zeros_like(weight_old), where=valid
    )
    # 1/ (sigma**2 + (sys_error_percentage*|A|)**2 )
    new_weight = np.divide(
        1.0,
        sigma_squared + (perc * abs(obs.vis.asnumpy())) ** 2,
        out=np.zeros_like(weight_old),
        where=valid,
    )

    return Observation(
        obs._antpos,
        obs._vis,
        new_weight,
        obs._polarization,
        obs._freq,
        obs._auxiliary_tables,
    )
