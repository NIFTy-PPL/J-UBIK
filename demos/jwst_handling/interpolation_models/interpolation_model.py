import nifty8.re as jft

from jax.experimental.sparse import BCOO

from typing import Callable, Union, Tuple, Optional
from numpy.typing import ArrayLike as ArrLike


def build_interpolation_model(
    interpolation: Callable[Union[ArrLike, Tuple[ArrLike, ArrLike]], ArrLike],
    sky: jft.Model,
    shift: Optional[jft.Model] = None
):
    if shift is None:
        return jft.Model(
            lambda x: interpolation(sky(x)),
            domain=jft.Vector(sky.domain)
        )

    domain = sky.domain.copy()
    domain.update(shift.domain)
    return jft.Model(
        lambda x: interpolation((sky(x), shift(x))),
        domain=jft.Vector(domain)
    )


def build_sparse_interpolation_model(
    sparse_matrix: BCOO,
    sky: jft.Model
):
    return jft.Model(lambda x: sparse_matrix @ sky(x).reshape(-1),
                     domain=jft.Vector(sky.domain))
