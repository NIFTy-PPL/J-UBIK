import numpy as np
from shapely.geometry import Polygon, box
from scipy.sparse import coo_matrix
from jax.experimental.sparse import BCOO

from numpy.typing import ArrayLike
from typing import Tuple


def get_nearest_index(coord: Tuple[float, float], grid: ArrayLike):
    '''Find the index of the nearest grid point to the given coordinate.'''
    dist_squared = (grid[0] - coord[0])**2 + (grid[1] - coord[1])**2
    return np.unravel_index(np.argmin(dist_squared), grid[0].shape)


class ValueCalculator:
    def __init__(self, coord_grid, points):
        self.triangle = Polygon([p for p in points])
        self.triangle_area = self.triangle.area
        self.hside = (abs(coord_grid[0][0, 1]-coord_grid[0][0, 0])/2,
                      abs(coord_grid[1][0, 0]-coord_grid[1][1, 0])/2)

        minx = min([p[0] for p in points])-self.hside[0]
        miny = min([p[1] for p in points])-self.hside[1]
        maxx = max([p[0] for p in points])+self.hside[0]
        maxy = max([p[1] for p in points])+self.hside[1]
        mask = ((coord_grid[0] >= minx) * (coord_grid[0] <= maxx) *
                (coord_grid[1] >= miny) * (coord_grid[1] <= maxy))
        self.masked_grid = coord_grid[0][mask], coord_grid[1][mask]

    def get_pixel_extrema(self, pix_cntr):
        minx, miny = pix_cntr[0]-self.hside[0], pix_cntr[1]-self.hside[1]
        maxx, maxy = pix_cntr[0]+self.hside[0], pix_cntr[1]+self.hside[1]
        return (minx, miny, maxx, maxy)

    def calculate_values(self, minimum=1e-11):
        values = {}
        for pix_cntr in zip(self.masked_grid[0], self.masked_grid[1]):
            minxy_maxxy = self.get_pixel_extrema(pix_cntr)
            pix_box = box(*minxy_maxxy)
            fractional_area = self.triangle.intersection(
                pix_box).area / self.triangle_area
            values[pix_cntr] = fractional_area
        return {k: v for k, v in values.items() if v > minimum}


def build_sparse_interpolation(
    index_grid: ArrayLike, edges: ArrayLike, mask: ArrayLike
):
    print('Calculating sparse interpolation matrix...')
    edges = edges[:, :, mask]

    data_length = edges[0, 0].size
    assert data_length == edges[0, 1].size
    assert data_length == edges[0, 0].shape[0]

    rows = []
    cols = []
    data = []

    for ii, pixel_edges in enumerate(edges.T):
        vc = ValueCalculator(index_grid, pixel_edges.T)
        values = vc.calculate_values()  # This also needs to be JAX compatible

        for (index_x, index_y), val in values.items():
            index_y, index_x = get_nearest_index(
                (index_x, index_y), index_grid)
            ind = np.ravel_multi_index((index_x, index_y), index_grid[0].shape)
            rows.append(ii)
            cols.append(ind)
            data.append(val)

    mat = coo_matrix(
        (data, (rows, cols)),
        shape=(data_length, index_grid[0].size))
    sparse_matrix = BCOO.from_scipy_sparse(mat)

    return sparse_matrix
