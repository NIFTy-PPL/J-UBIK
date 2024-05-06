import nifty8.re as jft

from jax.experimental.sparse import BCOO

from typing import Callable, Union, Tuple, Optional
from numpy.typing import ArrayLike as ArrLike


def build_integration_model(
    integration: Callable[Union[ArrLike, Tuple[ArrLike, ArrLike]], ArrLike],
    sky: jft.Model,
    sky_key: str,
    shift: Optional[jft.Model] = None
):
    # FIXME: Update such that the integration model takes a key (SKY_INTERNAL)
    # or index for the multifrequency model?

    domain = sky.target.copy()

    if shift is None:
        return jft.Model(
            lambda x: integration(x[sky_key]),
            domain=jft.Vector(domain)
        )

    domain.update(shift.domain)
    return jft.Model(
        lambda x: integration((x[sky_key], shift(x))),
        domain=jft.Vector(domain)
    )


def build_sparse_integration_model(
    sparse_matrix: BCOO,
    sky: jft.Model,
    sky_key: str,
):
    domain = sky.target

    return jft.Model(lambda x: sparse_matrix @ (x[sky_key]).reshape(-1),
                     domain=jft.Vector(domain))
