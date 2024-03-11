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


def find_extrema_indices(
    points: list,
    data_grid_wcs: gwcs
) -> tuple:

    index_points = [get_pixel(data_grid_wcs, p, tol=1e-1) for p in points]
    index_points = np.array([(pix[0].value, pix[1].value)
                            for pix in index_points])
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


for fltname, flt in config['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        dm = datamodels.open(filepath)

        data = dm.data
        wcs = dm.meta.wcs

        (minx, maxx, miny, maxy), dpix, (e00, e01, e10, e11) = find_extrema_indices(
            reconstruction_grid.world_extrema, wcs)

        plt.imshow(data[miny:maxy, minx:maxx], origin='lower',
                   norm=LogNorm(vmin=0.1, vmax=10))
        plt.show()

indcc = reconstruction_grid.wcs.world_to_pixel(dpix)
ind00 = reconstruction_grid.wcs.world_to_pixel(e00)
ind01 = reconstruction_grid.wcs.world_to_pixel(e01)
ind10 = reconstruction_grid.wcs.world_to_pixel(e10)
ind11 = reconstruction_grid.wcs.world_to_pixel(e11)

indc = np.array([indcc[ii] for ii in [0, 1]])
ind = np.array([[
    i[jj] for i in [ind00, ind01, ind10, ind11]
] for jj in [0, 1]])


mask = ((indcc[0] > 0) *
        (indcc[1] > 0) *
        (indcc[0] < reconstruction_grid.shape[0]) *
        (indcc[1] < reconstruction_grid.shape[1]))

d = data[miny:maxy, minx:maxx][mask]

indc = np.array([indcc[ii][mask] for ii in [0, 1]])
ind = np.array([[
    i[jj][mask] for i in [ind00, ind01, ind10, ind11]
] for jj in [0, 1]])


fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
axes[0].imshow(data[miny:maxy, minx:maxx], origin='lower',
               norm=LogNorm(vmin=0.1, vmax=10))
axes[1].imshow(mask, origin='lower')
plt.show()

ones = np.ones((reconstruction_grid.shape))
ones[::2, ::2] += 1
ones[1::2, ::2] += 2
ones[::2, 1::2] += 3

out = np.zeros((maxy-miny, maxx-minx))
out[mask] = d

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
axes[0].imshow(data[miny:maxy, minx:maxx], origin='lower',
               norm=LogNorm(vmin=0.1, vmax=10))
axes[1].imshow(out, origin='lower', norm=LogNorm(vmin=0.1, vmax=10))
axes[1].contour(mask)
axes[2].imshow(ones)
axes[2].scatter(ind[0], ind[1])
axes[2].scatter(ind[0, :, ii], ind[1, :,  ii])
axes[2].scatter(indc[0, ii], indc[1, ii])
plt.show()

for ii in range(5):
    plt.imshow(ones)
    plt.scatter(ind[0], ind[1])
    plt.scatter(ind[0, :, ii], ind[1, :,  ii])
    plt.scatter(indc[0, ii], indc[1, ii])
    plt.show()
