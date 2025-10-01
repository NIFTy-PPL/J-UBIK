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


from collections import defaultdict
from dataclasses import dataclass
from functools import partial, reduce, cached_property
from operator import add

import jax.numpy as jnp
import nifty.re as jft
from astropy import units as u
from nifty.cl.logger import logger
from nifty.re.likelihood import LikelihoodSum

from ....grid import Grid
from ....likelihood import connect_likelihood_to_model
from ..parse.data.data_loading import DataLoading
from ..parse.data.data_modify import ObservationModify
from ..parse.re.mosacing.beam_pattern import BeamPatternConfig
from ..parse.response import yaml_to_response_settings
from ..constants import RESOLVE_SPECTRAL_UNIT
from ..data.data_loading import load_and_modify_data_from_objects
from ..likelihood.mosaic_likelihood import (
    LikelihoodBuilder,
    build_likelihood_from_sky_beamer,
)
from ..mosaicing.sky_beamer import SkyBeamerJft, build_jft_sky_beamer
from ..multimessanger import (
    RadioSkyExtractor,
    build_radio_grid,
    build_radio_sky_extractor,
)
from ..telescopes.primary_beam import (
    build_primary_beam_pattern_from_beam_pattern_config,
)
from ..util import cast_to_dtype
from .mosaic_likelihood import build_likelihood_from_sky_beamer


@dataclass
class RadioLikelihoodProducts:
    """Container for managing and combining multiple likelihood builders.

    This class groups likelihood builders and provides functionality to combine
    them, either as a flat list or grouped by names. The combined likelihoods
    are then connected to a sky beam model.

    Attributes
    ----------
    likelihoods: List of LikelihoodBuilder objects to be combined.
    sky_beamer: Sky beam model used for connecting the final likelihood.
    radio_sky_extractor: Radio sky extraction tool (included for convenience).
    _names: Optional list of names for grouping likelihoods. When provided,
        likelihoods with the same name will be combined into a single
        LikelihoodSum object. Must have the same length as likelihoods.

    Raises
    ------
    AssertionError: If _names is provided but has a different length than
        the likelihoods list.
    """

    likelihoods: list[LikelihoodBuilder]
    sky_beamer: SkyBeamerJft
    radio_sky_extractor: RadioSkyExtractor  # NOTE : only for convenience
    _names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate that names and likelihoods have matching lengths if names are
        provided."""

        if self._names is not None:
            assert len(self._names) == len(self.likelihoods)

    def _get_likelihoods(self) -> list[jft.Likelihood]:
        """Get all likelihoods, potentially grouped by same names.

        Returns
        -------
        If no names are provided: A flat list of likelihood objects extracted
            from the likelihood builders.
        If names are provided: A list of LikelihoodSum objects, where each
            object contains all likelihoods that share the same name.
        """
        if self._names is None:
            return [t.likelihood for t in self.likelihoods]

        # Group by name
        grouped = defaultdict(list)
        for name, likelihood_obj in zip(self._names, self.likelihoods):
            grouped[name].append(likelihood_obj.likelihood)

        return [
            LikelihoodSum(*val, _key_template=key + "_{index}")
            for key, val in grouped.items()
        ]

    @cached_property
    def likelihood(self) -> jft.Likelihood:
        """Combine all likelihoods and connect them to the sky beam model.

        This property retrieves all likelihoods (potentially grouped by name),
        combines them using addition, and connects the resulting likelihood
        to the sky beam model.

        Returns
        -------
        A single Likelihood object that represents the sum of all likelihoods
        connected to the sky beam model.
        """

        likelihood = reduce(add, self._get_likelihoods())
        return connect_likelihood_to_model(likelihood, self.sky_beamer)


def build_radio_likelihood(
    data_names: list[str],
    cfg: dict,
    sky_grid: Grid,
    sky_domain: dict | jft.ShapeWithDtype,
    last_radio_bin: int | None,
    sky_unit: u.Unit | None = None,
    direction_key: str = "PHASE_DIR",
    data_key: str = "alma_data",
) -> RadioLikelihoodProducts:
    radio_sky_extractor = build_radio_sky_extractor(
        last_radio_bin,
        sky_domain=sky_domain,
        sky_unit=sky_unit,
        # transpose=response_settings.transpose,
    )
    radio_grid = build_radio_grid(last_radio_bin, sky_grid)

    response_backend_settings = yaml_to_response_settings(cfg["radio_response"])

    if not isinstance(data_names, list):
        assert isinstance(data_names, str)
        data_names = [data_names]

    likelihoods = []
    names = []
    sky_beamers = []
    for data_name in data_names:
        logger.info(f"\nLoading data: {data_name}")

        dl = DataLoading.from_yaml_dict(cfg[data_key][data_name])
        dm = ObservationModify.from_yaml_dict(cfg[data_key][data_name])
        observations = list(
            load_and_modify_data_from_objects(
                sky_frequencies=radio_grid.spectral,
                data_loading=dl,
                observation_modify=dm,
            )
        )

        # TODO: The following lines have to be simplified.
        beam_func = build_primary_beam_pattern_from_beam_pattern_config(
            BeamPatternConfig.from_yaml_dict(cfg[data_key][data_name]["dish"])
        )

        _sky_beamer = build_jft_sky_beamer(
            sky_shape_with_dtype=radio_sky_extractor.target,
            sky_fov=sky_grid.spatial.fov,
            sky_center=sky_grid.spatial.center,
            sky_frequency_binbounds=sky_grid.spectral.binbounds(
                RESOLVE_SPECTRAL_UNIT
            ).value,
            observations=observations,
            beam_func=beam_func,
            direction_key=direction_key,
            field_name_prefix=data_name,
        )
        sky_beamers.append(_sky_beamer)

        for field_name, beam_direction in _sky_beamer.beam_directions.items():
            for o in observations:
                if o.direction_from_key(direction_key) == beam_direction.direction:
                    names.append(data_name)
                    likelihoods.append(
                        build_likelihood_from_sky_beamer(
                            observation=o,
                            field_name=field_name,
                            sky_beamer=_sky_beamer,
                            sky_grid=sky_grid,
                            backend_settings=response_backend_settings,
                        )
                    )

        logger.info("")

    # Create the final sky beamer
    _sky_beamer = reduce(lambda x, y: x + y, sky_beamers)
    sky_beamer = jft.Model(
        lambda x: _sky_beamer(radio_sky_extractor(x)), domain=radio_sky_extractor.domain
    )

    return RadioLikelihoodProducts(
        likelihoods=likelihoods,
        sky_beamer=sky_beamer,
        radio_sky_extractor=radio_sky_extractor,
        _names=names,
    )
