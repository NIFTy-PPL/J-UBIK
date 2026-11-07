from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ..parse.rotation_and_shift.coordinates_correction import (
    MODEL_KEY,
    SHIFT_UNIT_KEY,
    ROTATION_UNIT_KEY,
    CoordinatesCorrectionPriorConfig,
)
from ..parse.zero_flux_model import match_band_key

DEFAULT_KEY = "default"


@dataclass
class FilterAlignment:
    filter_name: str
    boresight: list[SkyCoord] = field(default_factory=list)

    def load_correction_prior(self, raw: dict, number_of_observations: int):
        config = raw[match_band_key(raw, self.filter_name) or DEFAULT_KEY]

        config[MODEL_KEY] = raw[MODEL_KEY]
        config[SHIFT_UNIT_KEY] = raw[SHIFT_UNIT_KEY]
        config[ROTATION_UNIT_KEY] = raw[ROTATION_UNIT_KEY]

        return CoordinatesCorrectionPriorConfig.from_yaml_dict(
            raw=config,
            shift_shape=(number_of_observations, 2),
            rotation_shape=(number_of_observations, 1),
        )
