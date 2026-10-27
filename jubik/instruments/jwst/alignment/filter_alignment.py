from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ..parse.rotation_and_shift.coordinates_correction import (
    MODEL_KEY,
    OFFSET_KEY,
    SHIFT_UNIT_KEY,
    ROTATION_UNIT_KEY,
    CoordinatesCorrectionPriorConfig,
)


DEFAULT_KEY = "default"


@dataclass
class FilterAlignment:
    filter_name: str
    boresight: list[SkyCoord] = field(default_factory=list)

    def load_correction_prior(self, raw: dict, number_of_observations: int):
        if self.filter_name in raw:
            config = raw[self.filter_name]
        else:
            config = raw[DEFAULT_KEY]
        config = dict(config)  # `default` is shared by the filters, keep it unchanged

        config[MODEL_KEY] = raw[MODEL_KEY]
        config[SHIFT_UNIT_KEY] = raw[SHIFT_UNIT_KEY]
        config[ROTATION_UNIT_KEY] = raw[ROTATION_UNIT_KEY]
        config[OFFSET_KEY] = raw.get(OFFSET_KEY, None)

        return CoordinatesCorrectionPriorConfig.from_yaml_dict(
            raw=config,
            shift_shape=(number_of_observations, 2),
            rotation_shape=(number_of_observations, 1),
        )
