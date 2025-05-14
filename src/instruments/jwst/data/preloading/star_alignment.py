from ....gaia.star_finder import load_gaia_stars_in_fov
from ...alignment.filter_alignment import FilterAlignment
from ..jwst_data import JwstData


def star_alignment_preloading(filter_alignment: FilterAlignment, jwst_data: JwstData):
    filter_alignment.star_alignment.star_tables.append(
        load_gaia_stars_in_fov(
            jwst_data.wcs.world_corners(),
            filter_alignment.alignment_meta.library_path,
        )
    )
    return filter_alignment
