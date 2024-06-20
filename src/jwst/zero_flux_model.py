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


import nifty8.re as jft
from numpy.typing import ArrayLike

from .parametric_model import build_parametric_prior

ZERO_FLUX_KEY = 'zero_flux'


class ZeroFlux(jft.Model):
    def __init__(self, flux_prior: jft.Model, target: jft.ShapeWithDtype):
        self.flux_prior = flux_prior

        super().__init__(domain=self.flux_prior.domain, target=target)

    def __call__(self, params: dict, field: ArrayLike):
        return field + self.flux_prior(params)


def build_zero_flux_model(
    prefix: str,
    likelihood_config: dict,
    target_shape: jft.ShapeWithDtype
) -> jft.Model:
    model_cfg = likelihood_config.get(ZERO_FLUX_KEY, None)
    if model_cfg is None:
        return jft.Model(lambda x, y: y, domain=dict(), target=target_shape)

    prefix = '_'.join([prefix, ZERO_FLUX_KEY])

    shape = (1,)
    prior = build_parametric_prior(
        prefix, model_cfg['prior'], shape)
    prior_model = jft.Model(prior, domain={prefix: jft.ShapeWithDtype(shape)})

    return ZeroFlux(prior_model, target=target_shape)
