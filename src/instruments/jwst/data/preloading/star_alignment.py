from ....gaia.star_finder import load_gaia_stars_in_fov
from ...alignment.star_alignment import StarAlignment
from ..jwst_data import JwstData


def star_alignment_preloading(star_alignment: StarAlignment, jwst_data: JwstData):
    star_alignment.star_tables.append(
        load_gaia_stars_in_fov(
            jwst_data.wcs.world_corners(),
            star_alignment.alignment_meta.library_path,
        )
    )
