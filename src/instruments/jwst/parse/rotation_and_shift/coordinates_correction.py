# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from ..parametric_model.parametric_prior import (
    ProbabilityConfig, transform_setting_to_prior_config)

from astropy import units as u
from dataclasses import dataclass
from typing import Optional


ROTATION_UNIT_KEY = 'rotation_unit'
ROTATION_UNIT_DEFAULT = 'rad'
SHIFT_UNIT_KEY = 'shift_unit'
SHIFT_UNIT_DEFAULT = 'arcsec'
DEFAULT_KEY = 'default'

NOT_FILTER_KEYS = [
    ROTATION_UNIT_KEY,
    ROTATION_UNIT_DEFAULT,
    SHIFT_UNIT_KEY,
    SHIFT_UNIT_DEFAULT,
    DEFAULT_KEY,
]

SHIFT_KEY = 'shift'
ROTATION_KEY = 'rotation'


@dataclass
class CoordiantesCorrectionPriorConfig:
    shift: ProbabilityConfig
    rotation: ProbabilityConfig
    shift_unit: u.Unit
    rotation_unit: u.Unit

    def shift_in(self, unit: u.Unit):
        priors = self.shift.to_list()
        priors[1] = (priors[1]*self.shift_unit).to(unit).value
        if len(priors) == 4:
            priors[2] = (priors[2]*self.shift_unit).to(unit).value
        return transform_setting_to_prior_config(priors)

    def rotation_in(self, unit: u.Unit):
        priors = self.rotation.to_list()
        priors[1] = (priors[1]*self.rotation_unit).to(unit).value
        if len(priors) == 4:
            priors[2] = (priors[2]*self.rotation_unit).to(unit).value
        return transform_setting_to_prior_config(priors)


@dataclass
class CoordiantesCorrectionConfig:
    '''This class saves the `CoordiantesCorrectionPrior`s for the coordinate
    correction. Specifics can be provided for the different filters and their
    multiple datasets (dithers, etc.). If a specific filter gets a None, the
    coordinates from the data are fixed to the provided values.
    '''
    default: CoordiantesCorrectionPriorConfig
    filters: dict[str, list[Optional[CoordiantesCorrectionPriorConfig]]]

    def get_filter_or_default(self, filter_name: str, data_index: int) -> ProbabilityConfig:
        '''Returns the PriorConfig for the `filter_name` and `data_index` or the default.

        Parameters
        ----------
        filter_name : str
            The filter in question.
        '''
        filter_name = filter_name.lower()
        if filter_name in self.filters:
            for ii, coordiantes_correction_prior in enumerate(self.filters[filter_name]):
                if ii == data_index:
                    return coordiantes_correction_prior
        return self.default


def yaml_to_coordinates_correction_config(
    corrections_config: dict,
) -> dict:
    '''Parses the coordinate correction prior configuration.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing the rotation and shift priors.
    '''

    rotation_unit = getattr(
        u, corrections_config.get(ROTATION_UNIT_KEY, ROTATION_UNIT_DEFAULT))
    shift_unit = getattr(
        u, corrections_config.get(SHIFT_UNIT_KEY, SHIFT_UNIT_DEFAULT))

    default = CoordiantesCorrectionPriorConfig(
        shift=transform_setting_to_prior_config(
            corrections_config[DEFAULT_KEY][SHIFT_KEY]),
        shift_unit=shift_unit,
        rotation=transform_setting_to_prior_config(
            corrections_config[DEFAULT_KEY][ROTATION_KEY]),
        rotation_unit=rotation_unit,
    )

    filters = {}
    for filter_name, data_indices in corrections_config.items():
        filter_name = filter_name.lower()
        if filter_name in NOT_FILTER_KEYS:
            continue

        filter_prior_list = []
        for data_index, prior_settings in data_indices.items():
            filter_data_prior = CoordiantesCorrectionPriorConfig(
                shift=transform_setting_to_prior_config(
                    prior_settings[SHIFT_KEY]),
                shift_unit=shift_unit,
                rotation=transform_setting_to_prior_config(
                    prior_settings[ROTATION_KEY]),
                rotation_unit=rotation_unit,
            ) if prior_settings is not None else None
            filter_prior_list.append(filter_data_prior)

        filters[filter_name] = filter_prior_list

    return CoordiantesCorrectionConfig(default=default, filters=filters)
