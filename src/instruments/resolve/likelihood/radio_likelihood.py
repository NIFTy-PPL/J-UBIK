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
# Author: Julian Rüstig


from dataclasses import dataclass
from functools import partial, reduce

import jax.numpy as jnp
import nifty.re as jft
from astropy import units as u
from nifty.cl.logger import logger

from ....grid import Grid
from ....parse.instruments.resolve.data.data_loading import DataLoading
from ....parse.instruments.resolve.data.data_modify import ObservationModify
from ....parse.instruments.resolve.re.mosacing.beam_pattern import BeamPatternConfig
from ....parse.instruments.resolve.response import yaml_to_response_settings
from ....likelihood import connect_likelihood_to_model
from ..likelihood.mosaic_likelihood import build_likelihood_from_sky_beamer
from ..constants import RESOLVE_SPECTRAL_UNIT
from ..data.data_loading import load_and_modify_data_from_objects
from ..mosaicing.sky_beamer import SkyBeamerJft, build_jft_sky_beamer
from ..multimessanger import (
    RadioSkyExtractor,
    build_radio_grid,
    build_radio_sky_extractor,
)
from ..telescopes.primary_beam import (
    build_primary_beam_pattern_from_beam_pattern_config,
)
from .cast_to_dtype import cast_to_dtype

# TODO : This shouldn't depend on jwst. Hence, move this to higher level.
from ...jwst.parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)


@dataclass
class LikelihoodProducts:
    likelihoods: list[jft.Likelihood | jft.Gaussian]
    sky_beamer: SkyBeamerJft
    radio_sky_extractor: RadioSkyExtractor  # NOTE : only for convenience

    @property
    def likelihood(self) -> jft.Likelihood:
        # likelihoods = (t.likelihood for t in self.likelihoods)
        likelihood = reduce(lambda x, y: x + y, self.likelihoods)
        return connect_likelihood_to_model(likelihood, self.sky_beamer)


def build_radio_likelihood(
    data_names: str | list[str],
    cfg: dict,
    sky_grid: Grid,
    sky_model: jft.Model,
    last_radio_bin: int | None,
    sky_unit: u.Unit | None = None,
    direction_key: str = "PHASE_DIR",
) -> LikelihoodProducts:
    response_settings = yaml_to_response_settings(cfg["radio_response"])
    radio_sky_extractor = build_radio_sky_extractor(
        last_radio_bin,
        sky_model,
        sky_unit=sky_unit,
        transpose=response_settings.transpose,
    )
    radio_grid = build_radio_grid(last_radio_bin, sky_grid)

    if not isinstance(data_names, list):
        assert isinstance(data_names, str)
        data_names = [data_names]

    if "rotation_and_shift" in cfg["radio_response"]:
        coordinate_correction_config = CoordinatesCorrectionPriorConfig.from_yaml_dict(
            cfg["radio_response"]["rotation_and_shift"]
        )
    else:
        coordinate_correction_config = None

    likelihoods = []
    sky_beamers = []
    for data_name in data_names:
        logger.info(f"Loading data: {data_name}")

        dl = DataLoading.from_yaml_dict(cfg["alma_data"][data_name])
        dm = ObservationModify.from_yaml_dict(cfg["alma_data"][data_name])
        observations = list(
            load_and_modify_data_from_objects(
                sky_frequencies=radio_grid.spectral.binbounds_in(u.Unit("Hz")),
                data_loading=dl,
                observation_modify=dm,
            )
        )

        # TODO: The following lines have to be simplified.
        beam_func = build_primary_beam_pattern_from_beam_pattern_config(
            BeamPatternConfig.from_yaml_dict(cfg["alma_data"][data_name]["dish"])
        )

        _sky_beamer = build_jft_sky_beamer(
            sky_shape_with_dtype=radio_sky_extractor.target,
            sky_fov=sky_grid.spatial.fov,
            sky_center=sky_grid.spatial.center,
            sky_frequency_binbounds=sky_grid.spectral.binbounds_in(
                RESOLVE_SPECTRAL_UNIT
            ),
            observations=observations,
            beam_func=beam_func,
            direction_key=direction_key,
            field_name_prefix=data_name,
        )

        _likelihoods = []
        for field_name, beam_direction in _sky_beamer.beam_directions.items():
            for o in observations:
                if o.direction_from_key(direction_key) == beam_direction.direction:
                    _likelihoods.append(
                        build_likelihood_from_sky_beamer(
                            observation=o,
                            field_name=field_name,
                            sky_beamer=_sky_beamer,
                            sky_grid=sky_grid,
                            backend_settings=response_settings.backend,
                            cast_to_dtype=partial(cast_to_dtype, dtype=jnp.float32)
                            if o.is_single_precision()
                            else None,
                            phase_shift_correction_config=coordinate_correction_config,
                        )
                    )

        likelihood = reduce(lambda x, y: x + y, _likelihoods)

        likelihoods.append(likelihood)
        sky_beamers.append(_sky_beamer)
        logger.info("")

    # Create the final sky beamer
    _sky_beamer = reduce(lambda x, y: x + y, sky_beamers)
    sky_beamer = jft.Model(
        lambda x: _sky_beamer(radio_sky_extractor(x)), domain=radio_sky_extractor.domain
    )

    return LikelihoodProducts(
        likelihoods=likelihoods,
        sky_beamer=sky_beamer,
        radio_sky_extractor=radio_sky_extractor,
    )
