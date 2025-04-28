from ..parse.alignment.star_alignment import StarAlignment
from dataclasses import dataclass, field
from astropy.table import Table
from astropy.coordinates import SkyCoord
from ...gaia.star_finder import join_tables


@dataclass
class Source:
    id: int | list[int]
    position: SkyCoord

    def __getitem__(self, index: int):
        return Source(self.id[index], self.position[index])


@dataclass
class FilterAlignemnt:
    alignment_meta: StarAlignment
    source_tables: list[Table] = field(default_factory=list)

    def append_table(self, table: Table):
        self.source_tables.append(table)

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
