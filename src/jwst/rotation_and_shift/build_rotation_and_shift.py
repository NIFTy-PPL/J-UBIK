import nifty8.re as jft

from ..wcs.wcs_base import WcsBase
from ..wcs import (subsample_grid_centers_in_index_grid_non_vstack,
                   subsample_grid_corners_in_index_grid_non_vstack)
from ..reconstruction_grid import Grid
from .linear_rotation_and_shift import build_linear_rotation_and_shift
from .nufft_rotation_and_shift import build_nufft_rotation_and_shift
from .sparse_rotation_and_shift import build_sparse_rotation_and_shift
from .coordinates_correction import (
    build_coordinates_correction_model_from_grid, CoordinatesCorrection)

from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike
from typing import Callable, Union, Tuple, Optional


class RotationAndShiftModel(jft.Model):
    def __init__(
        self,
        sky_domain: dict,
        call: Callable,
        correction_model: Union[Callable, CoordinatesCorrection],
    ):
        assert isinstance(sky_domain, dict), ('Need to provide an internal key'
                                              'to the target of the sky model')

        self.sky_key = next(iter(sky_domain.keys()))
        self.correction_model = correction_model
        self.call = call

        correction_domain = correction_model.domain if isinstance(
            correction_model, CoordinatesCorrection) else {}
        super().__init__(domain=sky_domain | correction_domain)

    def __call__(self, x):
        return self.call(x[self.sky_key], self.correction_model(x))


def build_rotation_and_shift_model(
    sky_domain: dict,
    world_extrema: Tuple[SkyCoord],
    reconstruction_grid: Grid,
    data_grid_dvol: float,  # FIXME: should this be for each data pixel, i.e. an array?
    data_grid_wcs: WcsBase,
    model_type: str,
    subsample: int,
    kwargs: dict,
    coordinate_correction: Optional[dict] = None,
) -> Callable[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]], ArrayLike]:
    '''Rotation and shift model builder

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.

    world_extrema: Tuple[SkyCoord]
        The corners of the grid to be rotated and shifted into.

    reconstruction_grid: Grid
        The Grid underlying the reconstruction domain.

    data_grid_dvol: float
        The volume of the data pixel.

    data_grid_wcs: WcsBase
        The world coordinate system of the data grid.

    model_type: str
        The type of the rotation and shift model: (linear, nufft, sparse)

    subsample: int
        The subsample factor for the data grid. How many times a data pixel is
        subsampled in each direction.

    kwargs: dict
        linear:  dict, options
            - order: (0, 1), default: 1
            - sky_as_brightness: default: False
            - mode: wrap (default)

        sparse: dict, options
            - extend_factor, default: 1 (extension of the sky grid)
            - to_bottom_left: default: True (reconstruction in bottom left of extended grid)

        nufft: dict, options
            - sky_as_brightness: default: False

    coordinate_correction: dict
        domain_key: str
        priors: dict
            - shift: Mean and sigma for the Gaussian distribution of shift model.
            - rotation: Mean and sigma of the Gaussian distribution for theta [rad]


    Returns
    -------
    RotationAndShiftModel(dict(sky, correction)) -> rotated_and_shifted_sky
    '''

    assert reconstruction_grid.dvol.unit == data_grid_dvol.unit

    correction_model = build_coordinates_correction_model_from_grid(
        coordinate_correction['domain_key'],
        coordinate_correction['priors'],
        data_grid_wcs,
        reconstruction_grid,
        subsample_grid_centers_in_index_grid_non_vstack(
            world_extrema,
            data_grid_wcs,
            reconstruction_grid.wcs,
            subsample)
    )

    match model_type:
        case 'linear':
            linear_kwargs = kwargs.get(
                'linear',
                dict(order=1,
                     sky_as_brightness=False,
                     mode='wrap',)
            )
            call = build_linear_rotation_and_shift(
                sky_dvol=reconstruction_grid.dvol.value,
                sub_dvol=data_grid_dvol.value / subsample**2,
                **linear_kwargs,
            )

        case 'nufft':
            # FIXME: is the out_shape correct?
            out_shape = correction_model.target.shape[1:] if isinstance(
                correction_model, CoordinatesCorrection) else correction_model(None).shape[1:]

            call = build_nufft_rotation_and_shift(
                sky_dvol=reconstruction_grid.dvol.value,
                sub_dvol=data_grid_dvol.value / subsample**2,
                sky_shape=next(iter(sky_domain.values())).shape,
                out_shape=out_shape,
                **kwargs.get('nufft', dict(sky_as_brightness=False))
            )

        case 'sparse':
            # Sparse cannot update the coordinates, this is why the
            # correction_model is not passed to the builder.
            sparse_kwargs = kwargs.get('sparse', dict(
                extend_factor=1, to_bottom_left=False))
            call = build_sparse_rotation_and_shift(
                index_grid=reconstruction_grid.index_grid(**sparse_kwargs),
                subsample_corners=subsample_grid_corners_in_index_grid_non_vstack(
                    world_extrema,
                    data_grid_wcs,
                    reconstruction_grid.wcs,
                    subsample),
            )

            def correction_model(_): return None

        case _:
            raise NotImplementedError(
                f"{model_type} is not implemented. Available rotation_and_shift"
                f"methods are: (linear, nufft, sparse)"
            )

    return RotationAndShiftModel(
        sky_domain=sky_domain,
        call=call,
        correction_model=correction_model
    )
