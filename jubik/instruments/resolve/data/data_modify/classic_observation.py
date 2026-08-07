from ..observation import Observation
from .....polarization import PolarizationType

import resolve as rve


def convert_to_classic_observation(obs: Observation) -> rve.Observation:
    """Convert a jubik.resolve.Observation to a rve.Observation."""
    aux_table = {
        key: rve.AuxiliaryTable.from_list(val.to_list())
        for key, val in obs._auxiliary_tables.items()
    }
    polarization = obs.polarization
    if isinstance(polarization, PolarizationType):
        polarization = polarization.get_legacy_polarization()
    return rve.Observation(
        rve.AntennaPositions.from_list(obs.antenna_positions.to_list()),
        obs.vis.asnumpy(),
        obs.weight.asnumpy(),
        rve.Polarization.from_list(polarization.to_list()),
        obs.freq,
        aux_table,
    )
