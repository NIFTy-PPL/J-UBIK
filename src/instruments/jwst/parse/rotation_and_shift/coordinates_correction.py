# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from ..parametric_model.parametric_prior import (
    ProbabilityConfig,
    prior_config_factory,
)

from astropy import units as u
from dataclasses import dataclass, astuple


ROTATION_UNIT_KEY = "rotation_unit"
SHIFT_UNIT_KEY = "shift_unit"
SHIFT_KEY = "shift"
ROTATION_KEY = "rotation"


@dataclass
class CoordinatesCorrectionPriorConfig:
    shift: ProbabilityConfig
    shift_unit: u.Unit
    rotation: ProbabilityConfig
    rotation_unit: u.Unit

    def shift_in(self, unit: u.Unit):
        distribution, val1, val2, transformation = astuple(self.shift)
        val1 = (val1 * self.shift_unit).to(unit).value
        val2 = (val2 * self.shift_unit).to(unit).value
        return prior_config_factory([distribution, val1, val2, transformation])

    def rotation_in(self, unit: u.Unit):
        distribution, val1, val2, transformation = astuple(self.rotation)
        val1 = (val1 * self.rotation_unit).to(unit).value
        val2 = (val2 * self.rotation_unit).to(unit).value
        return prior_config_factory([distribution, val1, val2, transformation])

    @classmethod
    def from_yaml_dict(cls, raw: dict):
        return CoordinatesCorrectionPriorConfig(
            shift=prior_config_factory(raw[SHIFT_KEY]),
            rotation=prior_config_factory(raw[ROTATION_KEY]),
            shift_unit=getattr(u, raw[SHIFT_UNIT_KEY]),
            rotation_unit=getattr(u, raw[ROTATION_UNIT_KEY]),
        )
