from typing import Generator

from astropy.units import Quantity
from nifty.cl.logger import logger

from ....color import Color
from ..parse.data.data_loading import DataLoading, LoaderTemplate
from ..parse.data.data_modify import ObservationModify
from .data_modify import modify_observation
from .observation import Observation


def load_and_modify_data_from_objects(
    sky_frequencies: Quantity | Color,
    data_loading: DataLoading,
    observation_modify: ObservationModify,
) -> Generator[Observation, None, None]:
    """Load and modify the data, according to `data_loading`, and
    `observation_modify`.

    Parameters
    ----------
    sky_frequencies: list[float]
        The frequencies of the sky domain, according to which we can split the
        observation into frequency chunks.
    data_loading: DataLoading
        Model for the loading of the data, see DataLoading.
    observation_modify: ObservationModify
        Model for modifying the observations, ObservationModify.
    """

    if not isinstance(sky_frequencies, Color):
        sky_frequencies = Color(sky_frequencies)

    for ii, data_template in enumerate(data_loading.loader_templates):
        logger.info(f"\nLoading data: {data_template.file_path}")
        obs = Observation.load(data_template.file_path)
        yield modify_observation(sky_frequencies, obs, observation_modify(ii))
