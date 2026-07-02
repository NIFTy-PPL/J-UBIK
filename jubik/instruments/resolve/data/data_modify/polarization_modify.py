import numpy as np

from .....polarization import Polarization
from ...data.observation import Observation


def restrict_to_stokesi(observation: Observation):
    if observation.vis.domain[0].labels == ("I",):
        return observation
    # FIXME Do I need to change something in observation._auxiliary_tables?
    ind = observation._polarization.stokes_i_indices()
    vis = observation._vis[ind]
    wgt = observation._weight[ind]
    pol = observation._polarization.restrict_to_stokes_i()
    return Observation(
        observation._antpos,
        vis,
        wgt,
        pol,
        observation._freq,
        observation._auxiliary_tables,
    )


def restrict_to_polarization(observation: Observation, pol_label):
    # FIXME Do I need to change something in observation._auxiliary_tables?
    ind = observation.vis.domain[0].label2index(pol_label)
    vis = observation._vis[ind : ind + 1]
    wgt = observation._weight[ind : ind + 1]
    pol = Polarization([9])  # FIXME!!!!
    return Observation(
        observation._antpos,
        vis,
        wgt,
        pol,
        observation._freq,
        observation._auxiliary_tables,
    )


def average_stokesi(observation: Observation):
    # FIXME Do I need to change something in observation._auxiliary_tables?
    if observation.vis.domain[0].labels_eq("I"):
        return observation

    assert observation._vis.shape[0] == 2
    assert observation._polarization.restrict_to_stokes_i() == observation._polarization

    vis = np.sum(observation._weight * observation._vis, axis=0)[None]
    wgt = np.sum(observation._weight, axis=0)[None]
    invmask = wgt == 0.0
    vis /= wgt + np.ones_like(wgt) * invmask
    vis[invmask] = 0.0
    return Observation(
        observation._antpos,
        vis,
        wgt,
        Polarization.trivial(),
        observation._freq,
        observation._auxiliary_tables,
    )
