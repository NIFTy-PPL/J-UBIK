from ...data.observation import Observation


def restrict_to_autocorrelations(observation: Observation):
    slc = observation._antpos.ant1 == observation._antpos.ant2
    return observation[slc]


def remove_autocorrelations(observation: Observation):
    slc = observation._antpos.ant1 != observation._antpos.ant2
    return observation[slc]
