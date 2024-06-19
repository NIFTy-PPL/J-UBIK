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

LANCZOS_WINDOW = 3
ZERO_FLUX_KEY = 'zero_flux'


def build_zero_flux(prefix: str, likelihood_config: dict) -> jft.Model:
    model_cfg = likelihood_config.get(ZERO_FLUX_KEY, None)
    if model_cfg is None:
        return None

    prefix = '_'.join([prefix, ZERO_FLUX_KEY])
    zf = read_parametric_model(ZERO_FLUX_KEY)
    zfp = build_parametric_prior(zf, prefix, model_cfg)
    return jft.Model(
        lambda x: zf(zfp(x)),
        domain=zfp.domain
    )
