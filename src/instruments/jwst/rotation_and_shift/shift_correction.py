import nifty8.re as jft
import astropy.units as u

from ..parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)
from ..parametric_model import build_parametric_prior_from_prior_config


def build_shift_correction(
    domain_key: str,
    correction_config: CoordinatesCorrectionPriorConfig,
    unit: u.Unit | None = None,
) -> jft.Model:
    # Build shift prior
    shift_key = domain_key + "_shift"
    shift_shape = (2,)

    unit = correction_config.shift_unit if unit is None else unit

    return build_parametric_prior_from_prior_config(
        shift_key, correction_config.shift_in(unit), shift_shape, as_model=True
    )
