from ..observation import Observation

from nifty8.logger import logger


def restrict_and_average_to_stokes_i(obs: Observation):
    logger.info("Restricting and averaging to Stokes I")
    obs = obs.restrict_to_stokesi()
    obs = obs.average_stokesi()
    return obs
