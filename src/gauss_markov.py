#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-2-Clause

from functools import partial
from typing import Callable, Union, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, vmap
from jax.tree_util import tree_map

from nifty8.re.model import Initializer, LazyModel, Model
from nifty8.re.prior import LogNormalPrior, NormalPrior
from nifty8.re.tree_math import ShapeWithDtype, random_like


def _isscalar(x):
    return jnp.ndim(x) == 0



def nd_wiener_process(
        xi: Array,
        x0: Array,
        sigma: Union[float, Array],
        dt: Array,
):
    """Implements the Wiener process (WP)."""
    amp = jnp.sqrt(dt) * sigma * xi
    return jnp.cumsum(jnp.vstack((x0[None, ...], amp)), axis=0)


class NdGaussMarkovProcess(Model):
    def __init__(
        self,
        process: Callable,
        x0: Union[float, Array, LazyModel],
        dt: Union[float, Array],
        name='xi',
        N_steps: Optional[int] = None,
        **kwargs
    ):
        if _isscalar(dt):
            if N_steps is None:
                msg = "`N_steps` is None and `dt` is not a sequence"
                raise NotImplementedError(msg)
            dt = np.ones(N_steps) * dt
        shp = dt.shape + jnp.shape(
            x0.target if isinstance(x0, LazyModel) else x0
        )
        domain = {name: ShapeWithDtype(shp)}
        init = Initializer(
            tree_map(lambda x: partial(random_like, primals=x), domain)
        )
        if isinstance(x0, LazyModel):
            domain = domain | x0.domain
            init = init | x0.init
        self.x0 = x0
        for _, a in kwargs.items():
            if isinstance(a, LazyModel):
                domain = domain | a.domain
                init = init | a.init
        self.kwargs = kwargs
        self.name = name
        self.process = process
        slicing_tuple = (slice(None),) + (None,) * len(x0.shape)
        self.dt = jnp.array(dt)[slicing_tuple]

        super().__init__(domain=domain, init=init)

    def __call__(self, x):
        xi = x[self.name]
        xx = self.x0(x) if isinstance(self.x0, LazyModel) else self.x0
        tmp = {
            k: a(x) if isinstance(a, LazyModel) else a
            for k, a in self.kwargs.items()
        }
        return self.process(xi=xi, x0=xx, dt=self.dt, **tmp)


def build_wiener_process(
    x0: Union[tuple, float, Array, LazyModel],
    sigma: Union[tuple, float, Array, LazyModel],
    dt: Union[float, Array],
    name: str = 'wp',
    n_steps: int = None
) -> NdGaussMarkovProcess:
    """Implements the Wiener process (WP).

    The WP in continuous time takes the form:

    .. math::
        d/dt x_t = sigma xi_t ,

    where `xi_t` is continuous time white noise.

    Parameters:
    -----------
    x0: tuple, float, or LazyModel
        Initial position of the WP. Can be passed as a fixed value, or a
        generative Model. Passing a tuple is a shortcut to set a normal prior
        with mean and std equal to the first and second entry of the tuple
        respectively on `x0`.
    sigma: tuple, float, Array, LazyModel
        Standard deviation of the WP. Analogously to `x0` may also be passed on
        as a model. May also be passed as a sequence of length equal to `dt` in
        which case a different sigma is used for each time interval.
    dt: float or Array of float
        Step sizes of the process. In case it is a single float, `N_steps` must
        be provided to indicate the number of steps taken.
    name: str
        Name of the key corresponding to the parameters of the WP. Default `wp`.
    N_steps: int (optional)
        Option to set the number of steps in case `dt` is a scalar.

    Notes:
    ------
    In case `sigma` is time-dependent, i.E. passed on as a sequence
    of length equal to `xi`, it is assumed to be constant within each time bin,
    i.E. `sigma_t = sigma_i for t_i <= t < t_{i+1}`.
    """
    if isinstance(x0, tuple):
        x0 = NormalPrior(x0[0], x0[1], name=name + '_x0')
    if isinstance(sigma, tuple):
        sigma = LogNormalPrior(sigma[0], sigma[1], name=name + '_sigma')
    return NdGaussMarkovProcess(
        nd_wiener_process, x0, dt, name=name, N_steps=n_steps, sigma=sigma
    )


def build_fixed_point_wiener_process(
    x0: Union[tuple, float, Array, LazyModel],
    sigma: Union[tuple, float, Array, LazyModel],
    t: Array,
    reference_t_index: int,
    name: str = 'wp',
):
    """Implements the Wiener process (WP) with respect to a fixed point at the
    `reference_t_index`.

    The WP in continuous time takes the form:

    .. math::
        d/dt x_t = sigma xi_t ,

    where `xi_t` is continuous time white noise.

    Parameters:
    -----------
    x0: tuple, float, or LazyModel
        Initial position of the WP. Can be passed as a fixed value, or a
        generative Model. Passing a tuple is a shortcut to set a normal prior
        with mean and std equal to the first and second entry of the tuple
        respectively on `x0`.
    sigma: tuple, float, Array, LazyModel
        Standard deviation of the WP. Analogously to `x0` may also be passed on
        as a model. May also be passed as a sequence of length equal to `dt` in
        which case a different sigma is used for each time interval.
    t:  Array of float
        Time stamps of the process.
    reference_t_index: int
        Index of the reference time stamp in `t`.
    name: str
        Name of the key corresponding to the parameters of the WP.
        Default `wp`.

    Notes:
    ------
    In case `sigma` is time-dependent, i.E. passed on as a sequence
    of length equal to `xi`, it is assumed to be constant within each time bin,
    i.E. `sigma_t = sigma_i for t_i <= t < t_{i+1}`.
    """
    if isinstance(x0, tuple):
        x0 = NormalPrior(x0[0], x0[1], name=name + '_x0')
    if isinstance(sigma, tuple):
        sigma = LogNormalPrior(sigma[0], sigma[1], name=name + '_sigma')

    if reference_t_index == 0:
        dt = t[1:] - t[:-1]
        return build_wiener_process(x0, sigma, dt, name=name)

    elif reference_t_index == -1:
        # NOTE: this could also be covered by the next case.
        # However, this way has a faster execution time.
        dt = t[1:] - t[:-1]
        dt = dt[::-1]
        wp = build_wiener_process(x0, sigma, dt, name=name)
        def apply(x): return jnp.flip(wp(x), axis=0)
        return Model(apply, domain=wp.domain)

    else:
        dt = t[1:] - t[:-1]
        wp = build_wiener_process(x0, sigma, dt, name=name)

        def apply(x):
            xt = wp(x)
            return xt - xt[reference_t_index] + x0
        return Model(apply, domain=wp.domain)
