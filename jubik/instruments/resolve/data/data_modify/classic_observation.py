from astropy import units as u

from ...conventions import OBSERVATION_FREQ_UNIT, OBSERVATION_UVW_UNIT
from ..observation import Observation

# This compatibility adapter intentionally depends on classic RESOLVE: its output
# must use classic RESOLVE classes for consumption by that codebase. JuBIK's
# array-based observation classes cannot replace these output types.
import resolve as rve

CLASSIC_OBS_FREQ_UNIT = u.Hz
CLASSIC_OBS_UVW_UNIT = u.m


def convert_to_classic_observation(obs: Observation) -> rve.Observation:
    """Convert to classic RESOLVE, passing UVW in metres and frequency in Hz.

    JuBIK's bare arrays are interpreted using its observation conventions and
    converted explicitly to the fixed units expected by classic RESOLVE.
    """
    auxiliary_tables = (
        {} if obs._auxiliary_tables is None else obs._auxiliary_tables
    )
    aux_table = {
        key: rve.AuxiliaryTable.from_list(val.to_list())
        for key, val in auxiliary_tables.items()
    }
    # `obs.polarization` (a PolarizationType) only knows a subset of the legal
    # correlation orderings, hence use the loss-free legacy polarization.
    polarization = obs.legacy_polarization
    antenna_positions = obs.antenna_positions.to_list()
    antenna_positions[0] = (
        antenna_positions[0] * OBSERVATION_UVW_UNIT
    ).to_value(CLASSIC_OBS_UVW_UNIT)
    frequency = (obs.freq * OBSERVATION_FREQ_UNIT).to_value(CLASSIC_OBS_FREQ_UNIT)
    return rve.Observation(
        rve.AntennaPositions.from_list(antenna_positions),
        obs.vis,
        obs.weight,
        rve.Polarization.from_list(polarization.to_list()),
        frequency,
        aux_table,
    )
