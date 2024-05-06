from jwst import datamodels
from .wcs.wcs_jwst_data_model import WcsJwstDataModel

from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike


class JwstDataModel:
    def __init__(self, filepath: str):
        self.dm = datamodels.open(filepath)
        self.wcs = WcsJwstDataModel(self.dm.meta.wcs)
        self.shape = self.dm.data.shape

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
        minx, maxx, miny, maxy = self.wcs.index_from_wl_extrema(
            extrema, self.shape)
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
        minx, maxx, miny, maxy = self.wcs.index_from_wl_extrema(
            extrema, self.shape)
        return self.dm.err[miny:maxy, minx:maxx]
