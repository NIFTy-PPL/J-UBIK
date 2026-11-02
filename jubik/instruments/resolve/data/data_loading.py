from typing import Generator

from astropy.units import Quantity
from nifty.cl.logger import logger

from ....color import Color
from ..parse.data.data_loading import DataFile, DataLoading
from ..parse.data.data_modify import ObservationModify
from .data_modify import modify_observation
from .observation import Observation


def load_and_modify_data_from_objects(
    sky_frequencies: Quantity | Color,
    data_loading: DataLoading,
    observation_modify: ObservationModify,
) -> Generator[tuple[DataFile, Observation], None, None]:
    """Load and modify the data, according to `data_loading`, and
    `observation_modify`, and yield each observation with its `DataFile`.

    Parameters
    ----------
    sky_frequencies: Quantity | Color
        The frequencies of the sky domain, according to which we can split the
        observation into frequency chunks.
    data_loading: DataLoading
        Model for the loading of the data, see DataLoading.
    observation_modify: ObservationModify
        Model for modifying the observations, ObservationModify.

    Yields
    ------
    tuple[DataFile, Observation]
        Per loaded file, its `DataFile` and the modified observation, in the
        order of `DataLoading.data_files`.
    """

    if not isinstance(sky_frequencies, Color):
        sky_frequencies = Color(sky_frequencies)

    for ii, file in enumerate(data_loading.data_files):
        logger.info(f"\nLoading data: {file.path}")
        obs = Observation.load(file.path)
        obs = modify_observation(sky_frequencies, obs, observation_modify(ii))
        yield file, obs
