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
from typing import Callable, Union, Tuple


class RotationAndShiftModel(jft.Model):
    def __init__(
        self,
        sky_domain: dict,
        call: Callable,
        second_domain: dict = None
    ):
        self.sky_key = next(iter(sky_domain.keys()))

        need_sky_key = 'Need to provide an internal key to the target of the sky model'
        assert isinstance(sky_domain, dict), need_sky_key
        if second_domain is None:
            domain = sky_domain
            self.apply = lambda x: call(x[self.sky_key], None)

        else:
            domain = sky_domain | second_domain
            self.sec_key = next(iter(second_domain.keys()))
            self.apply = lambda x: call(x[self.sky_key], x[self.sec_key])

        super().__init__(domain=domain)

    def __call__(self, x):
        return self.apply(x)


def build_rotation_and_shift_model(
    sky_domain: dict,
    world_extrema: Tuple[SkyCoord],
    reconstruction_grid: Grid,
    data_grid_dvol: float,
    data_grid_wcs: WcsBase,
    model_type: str,
    subsample: int,
    kwargs_linear: dict,
    kwargs_sparse: dict,
    **kwargs
) -> Callable[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]], ArrayLike]:

    assert reconstruction_grid.dvol.unit == data_grid_dvol.unit

    parameters_center = dict(
        sky_dvol=reconstruction_grid.dvol.value,
        sub_dvol=data_grid_dvol.value / subsample**2,
        subsample_centers=subsample_grid_centers_in_index_grid_non_vstack(
            world_extrema,
            data_grid_wcs,
            reconstruction_grid.wcs,
            subsample),
        order=kwargs_linear['order']
    )
    parameters_corners = dict(
        index_grid=reconstruction_grid.index_grid(**kwargs_sparse),
        subsample_corners=subsample_grid_corners_in_index_grid_non_vstack(
            world_extrema,
            data_grid_wcs,
            reconstruction_grid.wcs,
            subsample),
    )

    MODELS = dict(
        linear=(build_linear_rotation_and_shift, parameters_center),
        nufft=(build_nufft_rotation_and_shift, parameters_center),
        sparse=(build_sparse_rotation_and_shift, parameters_corners)
    )

    if kwargs.get('updating', False):
        # from ..utils import build_shift_model
        raise NotImplementedError

    try:
        builder, params = MODELS[model_type.lower()]
    except KeyError as ke:
        raise NotImplementedError(
            f"{model_type} is not implemented. Available rotation_and_shift"
            f"methods are: {MODELS}"
        ) from ke

    call = builder(**params)

    return RotationAndShiftModel(sky_domain=sky_domain, call=call)
