# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2025 Max-Planck-Society
# Author: Vishal Johnson, Jakob Roth, Julian Rüstig, Andreas Popp

import nifty.re as jft

import operator
from functools import reduce
import jax
import jax.numpy as jnp

from typing import Iterable


class StokesAdder(jft.Model):
    """
    Converts four latent "pre-Stokes" fields into the physical Stokes
    parameters (I, Q, U, V).

    The latent fields (a, b, c, d) are transformed according to

        p = sqrt(b² + c² + d²)

        I = exp(a) cosh(p)

        Q = exp(a) sinh(p) * b / p

        U = exp(a) sinh(p) * c / p

        V = exp(a) sinh(p) * d / p

    This parameterization guarantees

        I >= 0

        I² >= Q² + U² + V²,

    which is equivalent to the brightness matrix being Hermitian and
    positive semidefinite.

    Parameters
    ----------
    correlated_fields : Sequence[jft.Model]
        Sequence of four models representing the latent fields

            (a, b, c, d)

        in this exact order. Each model must return an array of identical
        shape.

    Input
    -----
    x : PyTree
        Parameter tree passed unchanged to each latent field model.

    Output
    ------
    stokes : jax.Array
        Array whose first axis indexes the four Stokes parameters

            stokes[0] = I
            stokes[1] = Q
            stokes[2] = U
            stokes[3] = V
    """

    def __init__(
        self,
        pre_stokes_fields: Iterable[jft.Model],
    ):
        if len(self.psf) != 4:
            raise ValueError(
                f"Expected four pre-Stokes field models, got {len(pre_stokes_fields)}."
            )

        self._psf = pre_stokes_fields

        super().__init__(init=reduce(operator.or_, [model.init for model in self._psf]))

    def __call__(self, x):
        def get_stokes(pre_stokes):
            pol_int = jnp.sqrt(sum(pre_stokes[i] ** 2 for i in range(1, 4)))
            return jnp.concatenate(
                [
                    jnp.exp(pre_stokes[:1]) * jnp.cosh(pol_int),
                    (jnp.exp(pre_stokes[:1]) * jnp.sinh(pol_int) / pol_int)
                    * pre_stokes[1:],
                ]
            )

        pre_stokes = jnp.stack([model(x) for model in self._psf])
        dims_remaining = pre_stokes.shape[1:]
        pre_stokes = pre_stokes.reshape((4, -1))

        stokes = jax.vmap(get_stokes, in_axes=1, out_axes=-1)(pre_stokes)
        return stokes.reshape((4,) + dims_remaining)
