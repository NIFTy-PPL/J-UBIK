from .observation import Observation

import resolve as rve


def convert_to_classic_observation(obs: Observation) -> rve.Observation:
    """Convert a jubik.resolve.Observation to a rve.Observation."""
    aux_table = {
        key: rve.AuxiliaryTable.from_list(val.to_list())
        for key, val in obs._auxiliary_tables.items()
    }
    return rve.Observation(
        rve.AntennaPositions.from_list(obs.antenna_positions.to_list()),
        obs.vis.val.val,
        obs.weight.val.val,
        rve.Polarization.from_list(obs.polarization.to_list()),
        obs.freq,
        aux_table,
    )
