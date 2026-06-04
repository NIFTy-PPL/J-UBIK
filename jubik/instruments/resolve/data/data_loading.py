from astropy.units import Quantity
from nifty.cl.logger import logger

from ....color import Color
from ..parse.data.data_loading import DataLoading
from ..parse.data.data_modify import ObservationModify
from .data_modify import modify_observation
from .observation import Observation


def load_and_modify_data_from_objects(
    sky_frequencies: Quantity | Color,
    data_loading: DataLoading,
    observation_modify: ObservationModify,
):
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

    data_paths = [
        dt.format(field=fi, spw=spw)
        for fi in data_loading.field_ids
        for dt in data_loading.data_templates
        for spw in data_loading.spectral_windows
    ]

    for file in data_paths:
        logger.info(f"\nLoading data: {file}")
        obs = Observation.load(file)
        obs = modify_observation(sky_frequencies, obs, observation_modify)
        yield obs
