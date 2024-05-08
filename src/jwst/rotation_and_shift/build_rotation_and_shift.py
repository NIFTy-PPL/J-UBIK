from numpy.typing import ArrayLike
from typing import Callable, Union, Tuple

from .linear_rotation_and_shift import build_linear_rotation_and_shift
from .nufft_rotation_and_shift import build_nufft_rotation_and_shift


def build_rotation_and_shift_model(
    reconstruction_grid,
    data_grid,
    sky_model,
    model_type: str,
    subsample: int,
    **kwargs
) -> Callable[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]], ArrayLike]:

    sky_dvol = reconstruction_grid.dvol.value
    sub_dvol = data_grid.dvol.value / subsample**2,

    subsample_centers = subsample_grid_centers_in_index_grid(
        data_grid.world_extrema,
        data_grid.wcs,
        reconstruction_grid.wcs,
        subsample)

    # return build_linear_rotation_and_shift(
