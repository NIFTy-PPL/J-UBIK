import gwcs
from astropy.coordinates import SkyCoord


def get_pixel(data_wcs: gwcs.wcs, location: SkyCoord, tol=1e-7) -> tuple:
    return data_wcs.numerical_inverse(location, with_units=True, tolerance=tol)
