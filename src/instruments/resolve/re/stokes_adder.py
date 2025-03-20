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
# Author: Jakob Roth, Julian Rüstig


import operator
from functools import partial, reduce
import jax
import jax.numpy as jnp


class StokesAdder(jft.Model):
    def __init__(self, correlated_field_dict):
        """
        Takes in a dict for pre-stokes fields a,b,c,d and outputs the stokes fields I,Q,U,V.

        The relation is I = exp(a)*cosh(p), Q,U,V = exp(a)*sinh(p)* b/p,c/p,d/p with p = sqrt(b^2 + c^2 + d^2).
        """
        self.cfs = correlated_field_dict

        super().__init__(
            init=reduce(operator.or_, [value.init for value in self.cfs.values()])
        )

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

        pre_stokes = jnp.stack([cf(x) for cf in self.cfs.values()])
        dims_remaining = pre_stokes.shape[1:]
        pre_stokes = pre_stokes.reshape((4, -1))

        stokes = jax.vmap(get_stokes, in_axes=1, out_axes=-1)(pre_stokes)
        return stokes.reshape((4,) + dims_remaining)
