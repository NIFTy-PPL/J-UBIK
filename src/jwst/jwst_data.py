from jwst import datamodels
from .wcs.wcs_jwst_data import WcsJwstData

from .color import Color, ColorRange

from astropy.coordinates import SkyCoord
from astropy import units
from numpy.typing import ArrayLike
from numpy import isnan

# FIXME: NIRCam filter response can also be handled by the throughput curves: see,
nircam_filters = dict(
    # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers#gsc.tab=0
    # Pivot λ (μm), BW Δλ (μm), Effective response, Blue λ- (µm), Red λ+ (µm)
    F070W=(0.704,	0.128,	0.237,	0.624,	0.781),
    F090W=(0.901,	0.194,	0.318,	0.795,	1.005),
    F115W=(1.154,	0.225,	0.333,	1.013,	1.282),
    F140M=(1.404,	0.142,	0.434,	1.331,	1.479),
    F150W=(1.501,	0.318,	0.476,	1.331,	1.668),
    F162M=(1.626,	0.168,	0.469,	1.542,	1.713),
    F164N=(1.644,	0.020,	0.385,	1.635,	1.653),
    F150W2=(1.671,	1.227,	0.489,	1.007,	2.38),
    F182M=(1.845,	0.238,	0.505,	1.722,	1.968),
    F187N=(1.874,	0.024,	0.434,	1.863,	1.885),
    F200W=(1.990,	0.461,	0.525,	1.755,	2.227),
    F210M=(2.093,	0.205,	0.522,	1.992,	2.201),
    F212N=(2.120,	0.027,	0.420,	2.109,	2.134),
    F250M=(2.503,	0.181,	0.370,	2.412,	2.595),
    F277W=(2.786,	0.672,	0.412,	2.423,	3.132),
    F300M=(2.996,	0.318,	0.432,	2.831,	3.157),
    F322W3=(3.247,	1.339,	0.499,	2.432,	4.013),
    F323N=(3.237,	0.038,	0.290,	3.217,	3.255),
    F335M=(3.365,	0.347,	0.480,	3.177,	3.537),
    F356W=(3.563,	0.787,	0.530,	3.135,	3.981),
    F360M=(3.621,	0.372,	0.515,	3.426,	3.814),
    F405N=(4.055,	0.046,	0.418,	4.030,	4.076),
    F410M=(4.092,	0.436,	0.499,	3.866,	4.302),
    F430M=(4.280,	0.228,	0.526,	4.167,	4.398),
    F444W=(4.421,	1.024,	0.533,	3.881,	4.982),
    F460M=(4.624,	0.228,	0.460,	4.515,	4.747),
    F466N=(4.654,	0.054,	0.320,	4.629,	4.681),
    F470N=(4.707,	0.051,	0.316,	4.683,	4.733),
    F480M=(4.834,	0.303,	0.447,	4.662,	4.973),
)

miri_filters = dict(
    # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers#gsc.tab=0
    # Pivot λ (μm), BW Δλ (μm), Effective response, Blue λ- (µm), Red λ+ (µm)
    F560W=(5.635, 1.00, 0.245, 5.054, 6.171),
    F770W=(7.639, 1.95, 0.355, 6.581, 8.687),
    F1000W=(9.953, 1.80, 0.466, 9.023, 10.891),
    F1130W=(11.309, 0.73, 0.412, 10.953, 11.667),
    F1280W=(12.810, 2.47, 0.384, 11.588, 14.115),
    F1500W=(15.064, 2.92, 0.442, 13.527, 16.640),
    F1800W=(17.984, 2.95, 0.447, 16.519, 19.502),
    F2100W=(20.795, 4.58, 0.352, 18.477, 23.159),
    F2550W=(25.365, 3.67, 0.269, 23.301, 26.733),
    F2550WR=(25.365, 3.67, 0.269, 23.301, 26.733),
)

JWST_FILTERS = nircam_filters | miri_filters


def _get_dvol(filter: str):
    if filter in miri_filters:
        # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument#gsc.tab=0
        # https://iopscience.iop.org/article/10.1086/682254
        return (0.11*units.arcsec).to(units.deg)**2

    elif filter in nircam_filters:
        # https://jwst-docs.stsci.edu/jwst-near-infrared-camera#gsc.tab=0
        pivot = Color(nircam_filters[filter][0] * units.micrometer)

        if pivot in ColorRange(Color(0.6*units.micrometer),
                               Color(2.3*units.micrometer)):
            # 0.6–2.3 µm wavelength range
            return (0.031*units.arcsec).to(units.deg)**2

        else:
            # 2.4–5.0 µm wavelength range
            return (0.063*units.arcsec).to(units.deg)**2

    else:
        raise NotImplementedError('filter has to be in the supported filters'
                                  f'{JWST_FILTERS.keys()}')


class JwstData:
    def __init__(self, filepath: str):
        self.dm = datamodels.open(filepath)
        self.wcs = WcsJwstData(self.dm.meta.wcs)
        self.shape = self.dm.data.shape
        self.filter = self.dm.meta.instrument.filter.upper()
        self.camera = self.dm.meta.instrument.name.upper()
        self.dvol = _get_dvol(self.filter)

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

    def nan_inside_extrema(self, extrema: SkyCoord) -> ArrayLike:
        '''Get a nan-mask of the data inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        nan-mask : ArrayLike
            Mask corresponding to the nan values inside the extrema.

        '''
        minx, maxx, miny, maxy = self.wcs.index_from_wl_extrema(
            extrema, self.shape)
        return (
            (~isnan(self.dm.data[miny:maxy, minx:maxx])) *
            (~isnan(self.dm.err[miny:maxy, minx:maxx]))
        )

    @property
    def half_power_wavelength(self):
        pivot, bw, er, blue, red = JWST_FILTERS[self.filter]
        return ColorRange(Color(blue*units.micrometer),
                          Color(red*units.micrometer))

    @property
    def pivot_wavelength(self):
        pivot, *_ = JWST_FILTERS[self.filter]
        return Color(pivot * units.micrometer)

    @property
    def transmission(self):
        '''Effective response is the mean transmission value over the
        wavelength range.

        see:
        https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers#gsc.tab=0
        '''
        pivot, bw, effective_response, blue, red = JWST_FILTERS[self.filter]
        return effective_response
