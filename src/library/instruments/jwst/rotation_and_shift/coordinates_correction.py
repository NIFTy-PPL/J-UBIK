import jax.numpy as jnp
import nifty8.re as jft

from jax.numpy import array
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

from astropy import units as u

from ..parametric_model import build_parametric_prior
from ..wcs.wcs_jwst_data import WcsJwstData
from ..wcs.wcs_astropy import WcsAstropy
from ..reconstruction_grid import Grid


class CoordinatesCorrection(jft.Model):
    def __init__(
        self,
        shift_prior: jft.Model,
        rotation_prior: jft.Model,
        pix_distance: tuple[float],
        rotation_center: tuple[int, int],
        coords: ArrayLike,
    ):
        '''Rotation correction according to:
        ri = si Rot(theta) (pi - r)
            = Rot(theta) (si*pi -si*r),
        where si*r=rotation_center.
        '''
        self.rotation_center = rotation_center
        self.pix_distance = pix_distance
        self.shift_prior = shift_prior  # FIXME: Not prior to distribution_operator
        self.rotation_prior = rotation_prior
        self._coords = coords

        super().__init__(domain=rotation_prior.domain | shift_prior.domain)

    def __call__(self, params: dict) -> ArrayLike:
        shft = self.shift_prior(params) / self.pix_distance
        theta = self.rotation_prior(params)
        x = (jnp.cos(theta) * (self._coords[0]-self.rotation_center[0]) -
             jnp.sin(theta) * (self._coords[1]-self.rotation_center[1])
             ) + self.rotation_center[0] + shft[0]

        y = (jnp.sin(theta) * (self._coords[0]-self.rotation_center[0]) +
             jnp.cos(theta) * (self._coords[1]-self.rotation_center[1])
             ) + self.rotation_center[1] + shft[1]
        return jnp.array((x, y))

    # def _rotation(self, params: dict) -> ArrayLike:
    #     theta = self.rotation_prior(params)
    #     x = (jnp.cos(theta) * (self._coords[0]-self.rotation_center[0]) -
    #          jnp.sin(theta) * (self._coords[1]-self.rotation_center[1])
    #          ) + self.rotation_center[0]

    #     y = (jnp.sin(theta) * (self._coords[0]-self.rotation_center[0]) +
    #          jnp.cos(theta) * (self._coords[1]-self.rotation_center[1])
    #          ) + self.rotation_center[1]
    #     return jnp.array((x, y))

    # def _shift(self, params: dict) -> ArrayLike:
    #     shft = self.shift_prior(params) / self.pix_distance
    #     return self._coords + shft.reshape(2, 1, 1)


def build_coordinates_correction_model(
    domain_key: str,
    priors: Optional[dict],
    pix_distance: tuple[float],
    rotation_center: tuple[float, float],
    coords: ArrayLike,
) -> Union[Callable[[dict, ArrayLike], ArrayLike], CoordinatesCorrection]:
    '''The shift correction is a normal distribution on the shift: (x, y).
    (x, y) <- Gaussian(mean, sigma)

    Rotation correction is defined via:
    ri = si Rot(theta) (pi - r)
        = Rot(theta) (si*pi -si*r),
    where si*r=rotation_center given in units of the coordinates (si*pi).
    The prior distribution over theta is a normal distribution.

    Parameters
    ----------
    domain_key

    priors
        shift: Mean and sigma for the Gaussian distribution of shift model.
        rotation: Mean and sigma of the Gaussian distribution for theta [rad]

    pix_distance
        The si, i.e. the size of an individual pixel s.t. the shift is in units
        of the coords.

    rotation_center
        The rotation center, has to be relative to the coords which will be
        supplied to the RotationCorrection.

    coords
        The coordinates to be corrected
    '''
    if priors is None:
        return lambda _: coords

    # Build shift prior
    shift_key = domain_key + '_shift'
    shift_shape = (2,)
    pix_distance = array(pix_distance).reshape(shift_shape)
    shift_prior = build_parametric_prior(
        shift_key, priors['shift'], shift_shape)
    shift_prior_model = jft.Model(
        shift_prior, domain={shift_key: jft.ShapeWithDtype(shift_shape)})

    # Build rotation prior
    rotation_key = domain_key + '_rotation'
    rot_shape = (1,)
    rotation_prior = build_parametric_prior(
        rotation_key, priors['rotation'], rot_shape)
    rotation_prior_model = jft.Model(
        rotation_prior, domain={rotation_key: jft.ShapeWithDtype(rot_shape)})

    return CoordinatesCorrection(
        shift_prior_model,
        rotation_prior_model,
        pix_distance,
        rotation_center,
        coords,
    )


def build_coordinates_correction_model_from_grid(
    domain_key: str,
    priors: Optional[dict],
    data_wcs: Union[WcsJwstData, WcsAstropy],
    reconstruction_grid: Grid,
    coords: ArrayLike,
) -> Union[Callable[[dict, ArrayLike], ArrayLike], CoordinatesCorrection]:
    '''The shift correction is a normal distribution on the shift: (x, y).
    (x, y) <- Gaussian(mean, sigma)

    Rotation correction is defined via:
    ri = si Rot(theta) (pi - r)
        = Rot(theta) (si*pi -si*r),
    where si*r=rotation_center given in units of the coordinates (si*pi).
    The prior distribution over theta is a normal distribution.

    Parameters
    ----------
    domain_key

    priors
        - shift: Mean and sigma for the Gaussian distribution of shift model.
        - rotation: Mean and sigma of the Gaussian distribution for theta [rad]

    data_wcs

    reconstruction_grid

    coords
        The coordinates to be corrected
    '''

    if priors is None:
        return lambda _: coords

    if isinstance(data_wcs, WcsJwstData):
        header = data_wcs._wcs.to_fits()[0]
        rpix = (header['CRPIX1'],),  (header['CRPIX2'],)
        rpix = data_wcs.wl_from_index(rpix)
    elif isinstance(data_wcs, WcsAstropy):
        # FIXME: The following lines should be the same with the previous
        header = data_wcs._wcs.to_header()
        rpix = (header['CRPIX1'],  header['CRPIX2'])
        rpix = data_wcs.wl_from_index(rpix)[0]
    else:
        raise NotImplementedError(
            f"The type of world coordinate system {type(data_wcs)} is not "
            "supported. Supported types [WcsAstropy, WcsJwstData]."
        )

    rpix = reconstruction_grid.wcs.index_from_wl(rpix)[0]

    return build_coordinates_correction_model(
        domain_key=domain_key,
        priors=priors,
        pix_distance=[
            rd.to(u.arcsec).value for rd in reconstruction_grid.distances],
        rotation_center=rpix,
        coords=coords)
