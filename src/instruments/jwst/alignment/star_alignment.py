from ..parse.alignment.star_alignment import StarAlignment
from ..parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
    ROTATION_KEY,
    ROTATION_UNIT_KEY,
    SHIFT_KEY,
    SHIFT_UNIT_KEY,
)
from ..parse.parametric_model.parametric_prior import (
    ProbabilityConfig,
    prior_config_factory,
)

from dataclasses import dataclass, field
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from ...gaia.star_finder import join_tables

DEFAULT_KEY = "default"


@dataclass
class Source:
    id: int | list[int]
    position: SkyCoord

    def __getitem__(self, index: int):
        return Source(self.id[index], self.position[index])


@dataclass
class FilterAlignemnt:
    filter_name: str
    alignment_meta: StarAlignment
    correction_prior: CoordinatesCorrectionPriorConfig | None = None
    source_tables: list[Table] = field(default_factory=list)
    boresight: list[SkyCoord] = field(default_factory=list)

    def get_sources(self, data_id: int | None = None):
        if data_id is not None:
            table = self.source_tables[data_id]
        else:
            table = join_tables(self.source_tables)

        source_id = table["SOURCE_ID"]
        positions = SkyCoord(ra=table["ra"], dec=table["dec"], unit="deg")

        return Source(
            id=[
                id
                for id in source_id
                if id not in self.alignment_meta.exclude_source_id
            ],
            position=[
                position
                for id, position in zip(source_id, positions)
                if id not in self.alignment_meta.exclude_source_id
            ],
        )

    def load_correction_prior(self, raw: dict, number_of_observations: int):
        if self.filter_name in raw:
            config = raw[self.filter_name]
        else:
            config = raw[DEFAULT_KEY]

        self.correction_prior = CoordinatesCorrectionPriorConfig(
            shift=prior_config_factory(
                config[SHIFT_KEY], shape=(number_of_observations, 2)
            ),
            rotation=prior_config_factory(
                config[ROTATION_KEY], shape=(number_of_observations, 1)
            ),
            shift_unit=getattr(u, raw[SHIFT_UNIT_KEY]),
            rotation_unit=getattr(u, raw[ROTATION_UNIT_KEY]),
        )
