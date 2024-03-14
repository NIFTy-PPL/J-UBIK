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

import scipy.sparse as sp
from jax.experimental.sparse import BCOO
import jax.numpy as jnp

from jax.scipy.ndimage import map_coordinates

from jax import random
from jax import config
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')


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

    def index_grid(self, extend_factor=1) -> Tuple[ArrayLike, ArrayLike]:
        extent = [int(s * extend_factor) for s in self.shape]
        extent = [(e - s) // 2 for s, e in zip(self.shape, extent)]
        x, y = [np.arange(-e, s+e) for e, s in zip(extent, self.shape)]
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


config_path = 'JWST_config.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
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


def find_data_extrema(
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

    return (minx, maxx, miny, maxy)


def find_pixcenter_and_edges(
    minx: int, maxx: int, miny: int, maxy: int, data_grid_wcs: gwcs,
) -> tuple:

    pix_center = np.meshgrid(np.arange(minx, maxx, 1),
                             np.arange(miny, maxy, 1))
    e00 = pix_center - np.array([0.5, 0.5])[:, None, None]
    e01 = pix_center - np.array([0.5, -0.5])[:, None, None]
    e10 = pix_center - np.array([-0.5, 0.5])[:, None, None]
    e11 = pix_center - np.array([-0.5, -0.5])[:, None, None]

    pix_center, e00, e01, e10, e11 = [
        data_grid_wcs(*p, with_units=True) for p in [pix_center, e00, e01, e10, e11]]

    # FIXME: pix_center is making the mask

    return pix_center, (e00, e01, e10, e11)


def find_subsample_centers(
    minx: int, maxx: int, miny: int, maxy: int, subsample: int
) -> tuple:

    pix_center = np.array(np.meshgrid(np.arange(minx, maxx, 1),
                                      np.arange(miny, maxy, 1)))
    ps = np.arange(0.5/subsample, 1, 1/subsample) - 0.5
    ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)

    return ms[:, :, None, None] + pix_center


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


subsample = config['telescope']['integration_model']['subsample']
likelihoods = {}

for fltname, flt in config['files']['filter'].items():

    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        dm = datamodels.open(filepath)

        data = dm.data
        std = dm.err
        wcs = dm.meta.wcs

        (minx, maxx, miny, maxy) = find_data_extrema(
            reconstruction_grid.world_extrema, wcs, data.shape)

        pix_center, (e00, e01, e10, e11) = find_pixcenter_and_edges(
            minx, maxx, miny, maxy, wcs)

        subsample_centers = find_subsample_centers(
            minx, maxx, miny, maxy, subsample)
        subsample_centers = [wcs(*p, with_units=True)
                             for p in subsample_centers]
        ind_subsample_centers = np.array(
            [reconstruction_grid.wcs.world_to_pixel(
                p) for p in subsample_centers]
        )

        indcc, ind00, ind01, ind10, ind11 = [
            reconstruction_grid.wcs.world_to_pixel(p)
            for p in [pix_center, e00, e01, e10, e11]]
        index_centers = np.array(indcc)
        index_edges = np.array([ind00, ind01, ind11, ind10])

        mask = ((index_centers[0] > 0) *
                (index_centers[1] > 0) *
                (index_centers[0] < reconstruction_grid.shape[0]) *
                (index_centers[1] < reconstruction_grid.shape[1]) *
                ~np.isnan(data[miny:maxy, minx:maxx]))

        lh_name = f'{fltname}_{ii}'
        likelihoods[lh_name] = dict(
            mask=mask,
            index_edges=index_edges,
            index_subsample_centers=ind_subsample_centers,
            data=data[miny:maxy, minx:maxx],
            std=std[miny:maxy, minx:maxx],
        )


def build_sparse_interpolation(
        index_grid: ArrayLike, edges: ArrayLike, mask: ArrayLike):
    print('Calculating sparse interpolation matrix...')
    edges = edges[:, :, mask]

    data_length = edges[0, 0].size
    assert data_length == edges[0, 1].size
    assert data_length == edges[0, 0].shape[0]

    rows = []
    cols = []
    data = []

    for ii, pixel_edges in enumerate(edges.T):
        # Ensure this is JAX compatible
        vc = ValueCalculator(index_grid, pixel_edges.T)
        values = vc.calculate_values()  # This also needs to be JAX compatible

        for (index_x, index_y), val in values.items():
            ind = np.ravel_multi_index((index_x, index_y), index_grid[0].shape)
            rows.append(ii)
            cols.append(ind)
            data.append(val)

    coo_matrix = sp.coo_matrix(
        (data, (rows, cols)), shape=(data_length, index_grid[0].size))
    sparse_matrix = BCOO.from_scipy_sparse(coo_matrix)

    return sparse_matrix


def build_interpolation(subsample_centers, order=3):
    from functools import partial
    from jax import vmap, jit
    interpolation = partial(
        map_coordinates, order=order, mode='constant', cval=0.0)
    interpolation = vmap(interpolation, in_axes=(None, 0))

    def integration(x):
        out = interpolation(x, subsample_centers)
        return out.sum(axis=0) / out.shape[0]

    return jit(integration)


def build_updating_interpolation(subsample_centers, order=3):
    from functools import partial
    from jax import vmap, jit
    interpolation = partial(
        map_coordinates, order=order, mode='constant', cval=0.0)
    interpolation = vmap(interpolation, in_axes=(None, 0))

    def integration(x):
        field, xy_shift = x
        out = interpolation(
            field, subsample_centers - xy_shift[None, :, None, None])
        return out.sum(axis=0) / out.shape[0]

    return jit(integration)


for lh_name, lh in likelihoods.items():
    sparse_matrix = build_sparse_interpolation(
        reconstruction_grid.index_grid(), lh['index_edges'], lh['mask'])
    lh['sparse_matrix'] = sparse_matrix

    interpolation = build_interpolation(lh['index_subsample_centers'], order=1)
    lh['interpolation'] = interpolation

    updating_interpolation = build_updating_interpolation(
        lh['index_subsample_centers'], order=1)
    lh['updating_interpolation'] = updating_interpolation


if __name__ == '__main__':
    import jubik0 as ju
    import nifty8.re as jft

    def build_sparse_model(sparse_matrix, sky):
        return jft.Model(
            lambda x: sparse_matrix @ sky(x).reshape(-1),
            domain=jft.Vector(sky.domain))

    def build_interpolation_model(interpolation, mask, sky):
        return jft.Model(
            lambda x: interpolation(sky(x))[mask],
            domain=jft.Vector(sky.domain))

    def build_updating_interpolation_model(interpolation, mask, sky, shift):
        domain = sky.domain
        domain.update(shift.domain)
        return jft.Model(
            lambda x: interpolation((sky(x), shift(x)))[mask],
            domain=jft.Vector(domain))

    def build_shift_model(key, mean_sigma):
        from charm_lensing.models.parametric_models.parametric_prior import (
            build_prior_operator)
        distribution_model_key = ('normal', *mean_sigma)
        shape = (2,)

        shift_model = build_prior_operator(key, distribution_model_key, shape)
        domain = {key: jft.ShapeWithDtype((shape))}
        return jft.Model(shift_model, domain=domain)

    mock = False
    plots = True

    integration_model = config['telescope']['integration_model']['model']
    mean, sigma = (config['telescope']['integration_model']['mean'],
                   config['telescope']['integration_model']['sigma'])
    mean_sigma = (mean, sigma)

    key = random.PRNGKey(42)
    key, subkey, sampling_key, mini_par_key = random.split(key, 4)

    sky_dict = ju.create_sky_model_from_config(config_path)
    sky = sky_dict['sky']

    likes = []
    models = []
    for lh_key, lh in likelihoods.items():

        sparse_model = build_sparse_model(lh['sparse_matrix'], sky)
        lh['sparse_model'] = sparse_model

        lh['interpolation_model'] = build_interpolation_model(
            lh['interpolation'], lh['mask'], sky)

        shift = build_shift_model(lh_key+'shift', mean_sigma)
        lh['shift_model'] = shift
        lh['updating_interpolation_model'] = build_updating_interpolation_model(
            lh['updating_interpolation'], lh['mask'], sky, shift)

        if mock:
            mock_sky = sky(jft.random_like(sampling_key, sky.domain))
            data = sparse_matrix @ mock_sky.reshape(-1)
            scale = 0.1*data.mean()
            data += np.random.normal(scale=scale, size=data.shape)
            std = np.full(data.shape, scale)

            plot_data = np.zeros_like(lh['data'])
            mask = lh['mask']
            plot_data[mask] = data

        else:
            mask = lh['mask']
            data = jnp.array(lh['data'][mask], dtype=jnp.float64)
            std = jnp.array(lh['std'][mask], dtype=jnp.float64)
            plot_data = lh['data']

            pos = jft.random_like(sampling_key, sky.domain)
            tmp = np.zeros_like(plot_data)
            tmp[mask] = sparse_model(pos)

            tmp2 = np.zeros_like(plot_data)
            interpolation_model = lh['interpolation_model']
            tmp2[mask] = lh['interpolation_model'](pos)

            tmp3 = np.zeros_like(plot_data)
            pos3 = jft.random_like(
                mini_par_key, lh['updating_interpolation_model'].domain)
            shif = lh['shift_model'](pos3)
            tmp3[mask] = lh['updating_interpolation_model'](pos3)

            if plots:
                fig, axes = plt.subplots(1, 6)
                axes[0].imshow(plot_data, origin='lower', norm=LogNorm())
                axes[1].imshow(plot_data-tmp, origin='lower', norm=LogNorm())
                axes[2].imshow(tmp, origin='lower', norm=LogNorm())
                axes[3].imshow(tmp2, origin='lower', norm=LogNorm())
                axes[4].imshow(tmp-tmp2, origin='lower', cmap='RdBu_r')
                axes[5].imshow(tmp-tmp3, origin='lower', cmap='RdBu_r')
                plt.show()

        like = ju.library.likelihood.build_gaussian_likelihood(data, std)
        if integration_model in ['sparse']:
            like = like.amend(sparse_model, domain=sparse_model.domain)
        elif integration_model in ['interpolation']:
            like = like.amend(interpolation_model,
                              domain=interpolation_model.domain)
        elif integration_model in ['updating_interpolation']:
            like = like.amend(lh['updating_interpolation_model'],
                              domain=lh['updating_interpolation_model'].domain)

        likes.append(like)

    from functools import reduce
    like = reduce(lambda a, b: a + b, likes)

    cfg = ju.get_config(config_path)
    file_info = cfg['files']
    minimization_config = cfg['minimization']
    kl_solver_kwargs = minimization_config.pop('kl_kwargs')

    sky_dict.pop('target')

    def plot(s, x):
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 sky_dict,
                                 s,
                                 log_scale=True,
                                 relative_std=True,
                                 iteration=x.nit)

        from os.path import join
        from os import makedirs
        out_dir = join(file_info["res_dir"], 'residuals')
        makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(
            len(likelihoods.values()), 3, sharex=True, sharey=True,
            figsize=(9, 3 * len(likelihoods.values())),
            dpi=300)
        if len(likelihoods.values()) == 1:
            axes = np.array([axes])
        for ii, lh in enumerate(likelihoods.values()):
            sparse_model = lh['sparse_model']
            plot_data = lh['data']
            mask = lh['mask']

            arr = np.zeros_like(plot_data)
            arr[mask] = jft.mean([sparse_model(si) for si in s])
            im0 = axes[ii, 0].imshow(plot_data, origin='lower', norm=LogNorm())
            im1 = axes[ii, 1].imshow(arr, origin='lower', norm=LogNorm())
            im2 = axes[ii, 2].imshow((plot_data - arr)/lh['std'], origin='lower',
                                     cmap='RdBu_r', vmax=3, vmin=-3)
            for ax, im in zip(axes[ii], [im0, im1, im2]):
                fig.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/residuals_{x.nit}.png')

        if mock:
            sky_mean = jft.mean([sky(si) for si in s])
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            im0 = axes[0].imshow(mock_sky, origin='lower')
            im1 = axes[1].imshow(sky_mean, origin='lower')
            im2 = axes[2].imshow((mock_sky - sky_mean),
                                 origin='lower', cmap='RdBu_r')
            for ax, im in zip(axes, [im0, im1, im2]):
                fig.colorbar(im, ax=ax, shrink=0.7)
            plt.show()

        # ju.export_operator_output_to_fits(file_info["res_dir"],
        #                                   sky_dict,
        #                                   s,
        #                                   iteration=x.nit)
        # plot_simple_residuals(file_info["res_dir"], s, x.nit)

    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))
    samples, state = jft.optimize_kl(
        like,
        pos_init,
        key=key,
        kl_kwargs=kl_solver_kwargs,
        callback=plot,
        odir=file_info["res_dir"],
        **minimization_config)
