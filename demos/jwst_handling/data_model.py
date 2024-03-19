import numpy as np
from jwst import datamodels

from typing import Tuple, List
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike


class JwstDataModel:
    def __init__(self, filepath: str):
        self.dm = datamodels.open(filepath)
        self.wcs = self.dm.meta.wcs

    def data_extrema(self, extrema: SkyCoord) -> Tuple[int, int, int, int]:
        '''Find the minimum and maximum pixel coordinates of the data grid that
        contain the world location extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        minx, maxx, miny, maxy : Tuple[int, int, int, int]
            Minimum and maximum pixel coordinates of the data grid that contain
            the edge points.
        '''

        edges_dgrid = np.array([
            self.wcs.world_to_pixel(p) for p in extrema])

        check = (
            np.any(edges_dgrid < 0) or
            np.any(edges_dgrid >= self.dm.data.shape[0]) or
            np.any(edges_dgrid >= self.dm.data.shape[1])
        )
        if check:
            o = f"""One of the wcs extrema is outside the data grid
            {edges_dgrid}"""
            raise ValueError(o)

        minx = int(np.floor(edges_dgrid[:, 0].min()))
        maxx = int(np.ceil(edges_dgrid[:, 0].max()))
        miny = int(np.floor(edges_dgrid[:, 1].min()))
        maxy = int(np.ceil(edges_dgrid[:, 1].max()))
        return minx, maxx, miny, maxy

    def wl_pixelcenter_and_edges(
        self,
        extrema: SkyCoord
    ) -> Tuple[SkyCoord, Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord]]:
        '''Find the world locations of the pixel centers and edges of the data
        grid inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        pix_center : SkyCoord
            pixel centers of the data grid in global wcs

        e00, e01, e10, e11 : SkyCoord
            pixel edges of the data grid in global wcs

        '''
        minx, maxx, miny, maxy = self.data_extrema(extrema)

        pix_center = np.meshgrid(np.arange(minx, maxx, 1),
                                 np.arange(miny, maxy, 1))
        e00 = pix_center - np.array([0.5, 0.5])[:, None, None]
        e01 = pix_center - np.array([0.5, -0.5])[:, None, None]
        e10 = pix_center - np.array([-0.5, 0.5])[:, None, None]
        e11 = pix_center - np.array([-0.5, -0.5])[:, None, None]

        # Reminder: pix_center is used for the mask
        pix_center, e00, e01, e10, e11 = [
            self.wcs(*p, with_units=True) for p in
            [pix_center, e00, e01, e10, e11]
        ]

        return pix_center, (e00, e01, e10, e11)

    def wl_subsample_centers(
        self,
        extrema: SkyCoord,
        subsample: int
    ) -> List[SkyCoord]:
        '''Find the (subsampled) pixel centers for the location of pixels in
        a larger grid. The sub-part of the (subsampled) pixel centers is
        provided by the data_extrema argument.

        Parameters
        ----------
        data_extrema : tuple
            (minx, maxx, miny, maxy) inside the larger pixel grid

        subsample : int
            subsample factor of the pixel grid

        Returns
        -------
        pix_center : SkyCoord
            pixel centers of the data grid in global wcs

        '''
        minx, maxx, miny, maxy = self.data_extrema(extrema)

        pix_center = np.array(np.meshgrid(np.arange(minx, maxx, 1),
                                          np.arange(miny, maxy, 1)))
        ps = np.arange(0.5/subsample, 1, 1/subsample) - 0.5
        ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)
        subsample_centers = ms[:, :, None, None] + pix_center

        return [self.wcs(*p, with_units=True) for p in subsample_centers]

    def data_inside_extrema(self, extrema: SkyCoord) -> ArrayLike:
        '''Find the data values inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        data : ArrayLike
            Data values inside the extrema.

        '''
        minx, maxx, miny, maxy = self.data_extrema(extrema)
        return self.dm.data[miny:maxy, minx:maxx]

    def std_inside_extrema(self, extrema: SkyCoord) -> ArrayLike:
        '''Find the data values inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        data : ArrayLike
            Data values inside the extrema.

        '''
        minx, maxx, miny, maxy = self.data_extrema(extrema)
        return self.dm.err[miny:maxy, minx:maxx]
