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

from .parametric_model import build_parametric_prior

ZERO_FLUX_KEY = 'zero_flux'


def build_zero_flux_model(
    prefix: str,
    likelihood_config: dict,
) -> jft.Model:
    model_cfg = likelihood_config.get(ZERO_FLUX_KEY, None)
    if model_cfg is None:
        return jft.Model(lambda _: 0, domain=dict())

    prefix = '_'.join([prefix, ZERO_FLUX_KEY])

    shape = (1,)
    prior = build_parametric_prior(prefix, model_cfg['prior'], shape)
    return jft.Model(prior, domain={prefix: jft.ShapeWithDtype(shape)})
