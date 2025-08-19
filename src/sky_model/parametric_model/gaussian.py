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
# Author: Julian Ruestig


from typing import Iterable, Tuple

import jax.numpy as jnp
import nifty.re as jft
from jax import Array
from numpy.typing import NDArray


class Gaussian(jft.Model):
    def __init__(
        self,
        i0: jft.Model,
        center: jft.Model,
        covariance: jft.Model,
        off_diagonal: jft.Model,
        coordiantes: NDArray,
        log: bool = False,
    ) -> None:
        self.i0: jft.Model = i0
        self.center: jft.Model = center
        self.covariance: jft.Model = covariance
        self.off_diagonal: jft.Model = off_diagonal

        self._coordiantes = coordiantes
        self._log = log

        super().__init__(
            domain=i0.domain | center.domain | covariance.domain | off_diagonal.domain
        )

    def _log_call(self, x) -> Array:
        i0 = self.i0(x)
        center = self.center(x)
        sx, sy = self.covariance(x)
        theta = self.off_diagonal(x)

        x, y = self._coordiantes[0] - center[0], self._coordiantes[1] - center[1]

        a = jnp.cos(theta) ** 2 / (2 * sx**2) + jnp.sin(theta) ** 2 / (2 * sy**2)
        b = -jnp.sin(2 * theta) / (4 * sx**2) + jnp.sin(2 * theta) / (4 * sy**2)
        c = jnp.sin(theta) ** 2 / (2 * sx**2) + jnp.cos(theta) ** 2 / (2 * sy**2)

        return jnp.log(i0) - (a * x**2 + 2 * b * x * y + c * y**2)

    def _exp_call(self, x) -> Array:
        return jnp.exp(self._log_call(x))

    def __call__(self, x) -> Array:
        if self._log:
            return self._log_call(x)
        else:
            return self._exp_call(x)
