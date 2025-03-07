from jubik0.wcs.wcs_subsample_centers import (
    subsample_grid_centers_in_index_grid_non_vstack
)

import numpy as np


def grid_setup():
    from jubik0.parse.grid import GridModel
    from jubik0.grid import Grid

    grid_config = dict(
        sdim=384,
        fov=('2.0arcsec',)*2,
        coordinate_frame='icrs',
        sky_center=dict(ra='0.deg', dec='0.deg'),
        energy_unit='Hz',
        energy_bin=dict(e_min=[10e9], e_max=[12e9], reference_bin=0),
    )

    gm = GridModel.from_yaml_dict(grid_config)
    grid = Grid.from_grid_model(gm)

    return grid


def test_simple():
    grid = grid_setup()

    xx, yy = grid.spatial.index_grid_from_wl_extrema(
        grid.spatial.world_extrema())
    xxsub, yysub = subsample_grid_centers_in_index_grid_non_vstack(
        world_extrema=grid.spatial.world_extrema(),
        to_be_subsampled_grid_wcs=grid.spatial,
        index_grid_wcs=grid.spatial,
        subsample=1
    )
    xxsub, yysub = map(np.round, [xxsub, yysub])

    assert np.allclose(xx, xxsub)
    assert np.allclose(yy, yysub)
