from ..observation import Observation

import resolve as rve


def convert_to_classic_observation(obs: Observation) -> rve.Observation:
    """Convert a jubik.resolve.Observation to a rve.Observation."""
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
    return rve.Observation(
        rve.AntennaPositions.from_list(obs.antenna_positions.to_list()),
        obs.vis.asnumpy(),
        obs.weight.asnumpy(),
        rve.Polarization.from_list(polarization.to_list()),
        obs.freq,
        aux_table,
    )
