from .....parse.instruments.resolve.data.data_modify import WeightModify

from ..observation import Observation
from nifty8.logger import logger

import numpy as np


def weight_modify(obs: Observation, weight_modify: WeightModify | None):
    """Modify the weights of the observation. The weights get an added standard
    deviation of `weight_modify.percentage` of the amplitude of the
    visibilities.

    Parameters
    ----------
    obs: Observation
        The observation to modifiy
    weight_modify: WeightModify
        The parameters of the weight modify class, holding the percentage of
        the amplitude fraction.
    """

    if weight_modify is None:
        return obs

    logger.info(f"Weights modified by {weight_modify.percentage * 100} percent")

    weight_old = obs.weight.val
    perc = weight_modify.percentage

    # 1/ (sigma**2 + (sys_error_percentage*|A|)**2 )
    new_weight = 1 / ((1 / np.sqrt(weight_old)) ** 2 + (perc * abs(obs.vis.val)) ** 2)
    obs._weight = new_weight

    return obs
