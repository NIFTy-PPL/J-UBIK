from ..observation import Observation
from nifty8.logger import logger


def reverse_frequencies(obs: Observation) -> Observation:
    """This reverses the frequencies and returns an observation"""
    logger.info("Reverse frequencies")
    return Observation(
        obs.antenna_positions,
        obs.vis.val[:, :, ::-1],
        obs.weight.val[:, :, ::-1],
        obs.polarization,
        obs.freq[::-1],
        auxiliary_tables=obs._auxiliary_tables,
    )
