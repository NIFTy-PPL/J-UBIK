from ...data.observation import Observation


def _antenna_indices(observation: Observation, func_name: str):
    """Antenna indices of the observation, with a clear error if they are
    missing (imaging-only observations carry no calibration information)."""
    antpos = observation._antpos
    if antpos.only_imaging:
        raise ValueError(
            f"`{func_name}` needs antenna information (ant1/ant2), but the "
            "observation is imaging only, i.e. its AntennaPositions were "
            "created without antenna indices and times."
        )
    return antpos.ant1, antpos.ant2


def restrict_to_autocorrelations(observation: Observation):
    ant1, ant2 = _antenna_indices(observation, "restrict_to_autocorrelations")
    slc = ant1 == ant2
    return observation[slc]


def remove_autocorrelations(observation: Observation):
    ant1, ant2 = _antenna_indices(observation, "remove_autocorrelations")
    slc = ant1 != ant2
    return observation[slc]
