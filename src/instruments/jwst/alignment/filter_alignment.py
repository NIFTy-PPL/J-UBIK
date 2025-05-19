from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ..parse.alignment.star_alignment import StarAlignmentConfig
from ..parse.parametric_model.parametric_prior import (
    prior_config_factory,
)
from ..parse.rotation_and_shift.coordinates_correction import (
    MODEL_KEY,
    ROTATION_KEY,
    ROTATION_UNIT_KEY,
    SHIFT_KEY,
    SHIFT_UNIT_KEY,
    CoordinatesCorrectionPriorConfig,
)


DEFAULT_KEY = "default"


@dataclass
class FilterAlignment:
    filter_name: str
    correction_prior: CoordinatesCorrectionPriorConfig | None = None
    boresight: list[SkyCoord] = field(default_factory=list)

    def load_correction_prior(self, raw: dict, number_of_observations: int):
        if self.filter_name in raw:
            config = raw[self.filter_name]
        else:
            config = raw[DEFAULT_KEY]

        self.correction_prior = CoordinatesCorrectionPriorConfig(
            model=raw[MODEL_KEY],
            shift=prior_config_factory(
                config[SHIFT_KEY], shape=(number_of_observations, 2)
            ),
            rotation=prior_config_factory(
                config[ROTATION_KEY], shape=(number_of_observations, 1)
            ),
            shift_unit=getattr(u, raw[SHIFT_UNIT_KEY]),
            rotation_unit=getattr(u, raw[ROTATION_UNIT_KEY]),
        )
