from jubik0.wcs.wcs_subsample_centers import subsample_grid_centers_in_index_grid
from functools import partial

import numpy as np


def grid_setup():
    from jubik0.parse.grid import GridModel
    from jubik0.grid import Grid

    grid_config = dict(
        sdim=384,
        fov=("2.0arcsec",) * 2,
        coordinate_frame="icrs",
        sky_center=dict(ra="0.deg", dec="0.deg"),
        energy_unit="Hz",
        energy_bin=dict(e_min=[10e9], e_max=[12e9], reference_bin=0),
    )

    gm = GridModel.from_yaml_dict(grid_config)
    grid = Grid.from_grid_model(gm)

    return grid


def test_simple():
    grid = grid_setup()

    # Check subsample 1
    xx, yy = grid.spatial.bounding_box_index_grid_from_world_extrema(
        grid.spatial.world_corners(), indexing="xy"
    )
    xxsub, yysub = subsample_grid_centers_in_index_grid(
        world_corners=grid.spatial.world_corners(),
        to_be_subsampled_grid_wcs=grid.spatial,
        index_grid_wcs=grid.spatial,
        subsample=1,
        indexing="xy",
    )
    assert np.allclose(xx, xxsub, atol=1e-5)
    assert np.allclose(yy, yysub, atol=1e-5)

    # Check subsample 2
    xxsub, yysub = subsample_grid_centers_in_index_grid(
        world_corners=grid.spatial.world_corners(),
        to_be_subsampled_grid_wcs=grid.spatial,
        index_grid_wcs=grid.spatial,
        subsample=2,
        indexing="xy",
    )
    xxsub, yysub = map(partial(np.round, decimals=3), [xxsub, yysub])
    test_array = np.array(((-0.25, 0.25),) * 2)
    assert np.allclose(xxsub[:2, :2], test_array)
    assert np.allclose(yysub[:2, :2], test_array.T)

    # Check subsample 3
    xxsub, yysub = subsample_grid_centers_in_index_grid(
        world_corners=grid.spatial.world_corners(),
        to_be_subsampled_grid_wcs=grid.spatial,
        index_grid_wcs=grid.spatial,
        subsample=3,
        indexing="xy",
    )
    xxsub, yysub = map(partial(np.round, decimals=3), [xxsub, yysub])
    test_array = np.array(((-0.333, 0, 0.333),) * 3)
    assert np.allclose(xxsub[:3, :3], test_array)
    assert np.allclose(yysub[:3, :3], test_array.T)
