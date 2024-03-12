from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from jwst import datamodels
import numpy as np
import yaml

import astropy
# import webbpsf

from sys import exit

import gwcs
from astropy import units
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from typing import Tuple
from astropy.units import Unit
from numpy.typing import ArrayLike

from charm_lensing.spaces import get_xycoords

from shapely.geometry import Polygon, box


class Grid:
    def __init__(
        self,
        center: SkyCoord,
        shape: Tuple[int, int],
        fov: Tuple[Unit, Unit]
    ):
        self.center = center
        self.shape = shape
        self.fov = fov
        self.wcs = self._get_wcs()

    def _get_wcs(self) -> WCS:
        return get_coordinate_system(
            self.center,
            self.shape,
            (self.fov[0].to(units.deg), self.fov[1].to(units.deg))
        )

    @property
    def world_extrema(self) -> ArrayLike:
        return [self.wcs.array_index_to_world(*(0, 0)),
                self.wcs.array_index_to_world(*(self.shape[0], 0)),
                self.wcs.array_index_to_world(*(0, self.shape[1])),
                self.wcs.array_index_to_world(*(self.shape[0], self.shape[1]))]

    def relative_coords(self, unit="arcsec") -> ArrayLike:
        unit = getattr(units, unit)
        distances = [f.to(unit)/s for f, s in zip(self.fov, self.shape)]
        return get_xycoords(self.shape, distances)

    def index_grid(self) -> Tuple[ArrayLike, ArrayLike]:
        x, y = np.arange(self.shape[0]), np.arange(self.shape[1])
        return np.meshgrid(x, y, indexing='xy')


def get_coordinate_system(
    center: SkyCoord,
    shape: Tuple[int, int],
    fov: Tuple[Unit, Unit]
) -> WCS:
    # Create a WCS object
    w = WCS(naxis=2)

    # Set up ICRS system
    w.wcs.crpix = [shape[0] / 2, shape[1] / 2]
    w.wcs.cdelt = [-fov[0].to(units.deg).value / shape[0],
                   fov[1].to(units.deg).value / shape[1]]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return w


def define_location(config: dict) -> SkyCoord:
    ra = config['telescope']['pointing']['ra']
    dec = config['telescope']['pointing']['dec']
    frame = config['telescope']['pointing'].get('frame', 'icrs')
    unit = getattr(units, config['telescope']['pointing'].get('unit', 'deg'))

    return SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)


def get_pixel(wcs: gwcs.wcs, location: SkyCoord, tol=1e-7) -> tuple:
    return wcs.numerical_inverse(location, with_units=True, tolerance=tol)


config = yaml.load(open('JWST_config.yaml', 'r'), Loader=yaml.SafeLoader)
WORLD_LOCATION = define_location(config)
FOV = config['telescope']['fov'] * \
    getattr(units, config['telescope'].get('fov_unit', 'arcsec'))
shape = (config['grid']['npix'], config['grid']['npix'])


# defining the reconstruction grid
reconstruction_grid = Grid(
    WORLD_LOCATION,
    shape,
    (FOV.to(units.deg), FOV.to(units.deg))
)


def find_extrema_and_indices(
    points: list,
    data_grid_wcs: gwcs,
    data_shape: Tuple[int, int]
) -> tuple:

    # index_points = [get_pixel(data_grid_wcs, p, tol=1e-4) for p in points]
    # index_points = np.array([(pix[0].value, pix[1].value)
    #                         for pix in index_points])
    index_points = np.array(
        [data_grid_wcs.world_to_pixel(p) for p in points])

    check = (
        np.any(index_points < 0) or
        np.any(index_points >= data_shape[0]) or
        np.any(index_points >= data_shape[1])
    )
    if check:
        raise ValueError(
            f"One of the points is outside the grid \n{index_points}")

    minx = int(np.floor(index_points[:, 0].min()))
    maxx = int(np.ceil(index_points[:, 0].max()))
    miny = int(np.floor(index_points[:, 1].min()))
    maxy = int(np.ceil(index_points[:, 1].max()))

    dpix = np.meshgrid(np.arange(minx, maxx, 1), np.arange(miny, maxy, 1))
    e00 = dpix - np.array([0.5, 0.5])[:, None, None]
    e01 = dpix - np.array([0.5, -0.5])[:, None, None]
    e10 = dpix - np.array([-0.5, 0.5])[:, None, None]
    e11 = dpix - np.array([-0.5, -0.5])[:, None, None]

    dpix, e00, e01, e10, e11 = [wcs(*p, with_units=True)
                                for p in [dpix, e00, e01, e10, e11]]

    return (minx, maxx, miny, maxy), dpix, (e00, e01, e10, e11)


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

        values = {k: v for k, v in values.items() if v > minimum}

        return values


def check_plot(index_edges, kk=0):

    ones = np.zeros((reconstruction_grid.shape))
    ones[::2, ::2] += 1
    ones[1::2, ::2] += 2
    ones[::2, 1::2] += 3

    out = np.zeros((maxy-miny, maxx-minx))
    out[mask] = data[miny:maxy, minx:maxx][mask]

    fig, axes = plt.subplots(1, 3)
    fig.suptitle(f'{fltname}', fontsize=16)
    axes[0].imshow(data[miny:maxy, minx:maxx], origin='lower',
                   norm=LogNorm())
    axes[1].imshow(out, origin='lower', norm=LogNorm())
    axes[1].contour(mask)
    axes[2].imshow(ones, origin='lower')
    axes[2].scatter(index_edges[:, 1], index_edges[:, 0], s=2, c='r')
    axes[2].scatter(index_centers[1], index_centers[0])
    axes[2].scatter(index_edges[:, 1, kk], index_edges[:, 0, kk])
    plt.show()


def check_index(index_edges, N=5):
    for kk in range(N):
        vc = ValueCalculator(
            reconstruction_grid.index_grid(), index_edges[:, :, kk])
        values = vc.calculate_values()

        fig, axes = plt.subplots(1, 1)
        fig.suptitle(f'{fltname}', fontsize=16)

        ones = np.zeros((reconstruction_grid.shape))
        for index, val in values.items():
            ones[index] = val

        axes.imshow(ones, origin='lower',)
        axes.scatter(index_edges[:, 1], index_edges[:, 0], s=2, c='r')
        axes.scatter(index_centers[1], index_centers[0])
        axes.scatter(index_edges[:, 1, kk], index_edges[:, 0, kk])
        plt.show()


def calculate_interpolation(
    index_grid: ArrayLike,
    edges: ArrayLike
):
    pass


for fltname, flt in config['files']['filter'].items():

    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        dm = datamodels.open(filepath)

        data = dm.data
        wcs = dm.meta.wcs

        (minx, maxx, miny, maxy), dpix, (e00, e01, e10, e11) = \
            find_extrema_and_indices(
                reconstruction_grid.world_extrema, wcs, data.shape)

        indcc, ind00, ind01, ind10, ind11 = [
            reconstruction_grid.wcs.world_to_pixel(p)
            for p in [dpix, e00, e01, e10, e11]]
        index_centers = np.array(indcc)
        index_edges = np.array([ind00, ind01, ind11, ind10])

        mask = ((index_centers[0] > 0) *
                (index_centers[1] > 0) *
                (index_centers[0] < reconstruction_grid.shape[0]) *
                (index_centers[1] < reconstruction_grid.shape[1]))

        index_centers = index_centers[:, mask]
        index_edges = index_edges[:, :, mask]

        check_plot(index_edges, kk=0)
        check_index(index_edges, N=2)

        exit()
