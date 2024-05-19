import nifty8.re as jft

from ..wcs.wcs_base import WcsBase
from ..wcs import (subsample_grid_centers_in_index_grid_non_vstack,
                   subsample_grid_corners_in_index_grid_non_vstack)
from ..reconstruction_grid import Grid
from .linear_rotation_and_shift import build_linear_rotation_and_shift
from .nufft_rotation_and_shift import build_nufft_rotation_and_shift
from .sparse_rotation_and_shift import build_sparse_rotation_and_shift


from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike
from typing import Callable, Union, Tuple, Optional


class RotationAndShiftModel(jft.Model):
    def __init__(
        self,
        sky_domain: dict,
        call: Callable,
        correction_domain: Optional[dict] = None
    ):
        need_sky_key = 'Need to provide an internal key to the target of the sky model'
        assert isinstance(sky_domain, dict), need_sky_key

        self.sky_key = next(iter(sky_domain.keys()))

        if correction_domain is None:
            domain = sky_domain
            self.apply = lambda x: call(x[self.sky_key], None)

        else:
            domain = sky_domain | correction_domain
            self.sec_key = next(iter(correction_domain.keys()))
            self.apply = lambda x: call(x[self.sky_key], x[self.sec_key])

        super().__init__(domain=domain)

    def __call__(self, x):
        return self.apply(x)


def build_rotation_and_shift_model(
    sky_domain: dict,
    world_extrema: Tuple[SkyCoord],
    reconstruction_grid: Grid,
    data_grid_dvol: float,  # FIXME: should this be for each data pixel, i.e. an array?
    data_grid_wcs: WcsBase,
    model_type: str,
    subsample: int,
    kwargs: dict,
    shift_and_rotation_correction_domain: Optional[dict] = None,
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

        sparse: dict, options
            - extend_factor, default: 1 (extension of the sky grid)
            - to_bottom_left: default: True (reconstruction in bottom left of extended grid)

        nufft: dict, options
            - sky_as_brightness: default: False

    shift_and_rotation_correction_domain: Optional[dict]
        The target domain of the shift and rotation correction model:
        {key: ShapeDtype}

    Returns
    -------
    RotationAndShiftModel(dict(sky, correction)) -> rotated_and_shifted_sky
    '''

    assert reconstruction_grid.dvol.unit == data_grid_dvol.unit

    match model_type:

        case 'linear':
            linear_kwargs = kwargs.get('linear', dict(
                order=1, sky_as_brightness=False))
            linear_kwargs['updating'] = (
                False if shift_and_rotation_correction_domain is None else True
            )
            call = build_linear_rotation_and_shift(
                sky_dvol=reconstruction_grid.dvol.value,
                sub_dvol=data_grid_dvol.value / subsample**2,
                subsample_centers=subsample_grid_centers_in_index_grid_non_vstack(
                    world_extrema,
                    data_grid_wcs,
                    reconstruction_grid.wcs,
                    subsample),
                **linear_kwargs
            )

        case 'nufft':
            nufft_kwargs = kwargs.get('nufft', dict(sky_as_brightness=False))
            call = build_nufft_rotation_and_shift(
                sky_dvol=reconstruction_grid.dvol.value,
                sub_dvol=data_grid_dvol.value / subsample**2,
                subsample_centers=subsample_grid_centers_in_index_grid_non_vstack(
                    world_extrema,
                    data_grid_wcs,
                    reconstruction_grid.wcs,
                    subsample),
                sky_shape=next(iter(sky_domain.values())).shape,
                **nufft_kwargs
            )

        case 'sparse':
            sparse_kwargs = kwargs.get('sparse', dict(
                extend_factor=1, to_bottom_left=True))
            call = build_sparse_rotation_and_shift(
                index_grid=reconstruction_grid.index_grid(**sparse_kwargs),
                subsample_corners=subsample_grid_corners_in_index_grid_non_vstack(
                    world_extrema,
                    data_grid_wcs,
                    reconstruction_grid.wcs,
                    subsample),
            )

        case _:
            raise NotImplementedError(
                f"{model_type} is not implemented. Available rotation_and_shift"
                f"methods are: (linear, nufft, sparse)"
            )

    return RotationAndShiftModel(
        sky_domain=sky_domain,
        call=call,
        correction_domain=shift_and_rotation_correction_domain
    )
