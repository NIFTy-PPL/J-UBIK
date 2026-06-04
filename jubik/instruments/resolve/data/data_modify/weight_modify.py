import numpy as np
from nifty.cl.logger import logger

from ...parse.data.data_modify.modify_weight import SystematicErrorBudget
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

    # 1/ (sigma**2 + (sys_error_percentage*|A|)**2 )
    new_weight = 1 / (
        # (1 / np.sqrt(weight_old)) ** 2 + (perc * abs(obs.vis.asnumpy())) ** 2
        (1 / weight_old) + (perc * abs(obs.vis.asnumpy())) ** 2
    )
    obs._weight = new_weight

    return obs
