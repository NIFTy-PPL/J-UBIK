# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2021-2023 Technical University Munich
# Copyright(C) 2022-2023 Philipp Arras
# Author: Philipp Arras, Jakob Knollmüller

import numpy as np
import resolve as rve


def load_uvfits(file_name, polarization="stokes"):
    import ehtim

    pol = "stokes" if polarization == "stokesI" else polarization
    obs = ehtim.obsdata.load_uvfits(file_name, polrep=pol)
    unique_antennas = obs.tarr["site"]
    antdct = {aa: ii for ii, aa in enumerate(unique_antennas)}
    ant1 = np.array([antdct[aa] for aa in obs.data["t1"]])
    ant2 = np.array([antdct[aa] for aa in obs.data["t2"]])
    position = np.vstack([obs.tarr["x"], obs.tarr["y"], obs.tarr["z"]]).T
    anttbl = rve.AuxiliaryTable({"POSITION": position, "STATION": unique_antennas})
    if obs.polrep == "circ":
        vis = []
        sigma = []
        # for kk in ["qvis", "uvis", "vvis", "qsigma", "usigma", "vsigma"]:
        for kk in ["rr", "rl", "lr", "ll"]:
            vis.append(obs.data[f"{kk}vis"].reshape(-1, 1))
            sigma.append(
                np.nan_to_num(obs.data[f"{kk}sigma"].reshape(-1, 1), nan=np.inf)
            )
        vis = np.array(vis)
        sigma = np.array(sigma)
        pol = rve.Polarization((5, 6, 7, 8))

    elif polarization == "stokesI":
        vis = obs.data["vis"].reshape(1, -1, 1)
        sigma = obs.data["sigma"].reshape(1, -1, 1)
        pol = rve.Polarization.trivial()
    else:
        raise NotImplementedError
    freq = np.array([obs.rf])
    time = obs.data["time"] * 3600
    uvw = np.vstack([obs.data["u"], obs.data["v"], 0.0 * obs.data["u"]]).T
    uvw *= rve.SPEEDOFLIGHT / obs.rf
    antpos = rve.AntennaPositions(uvw, ant1, ant2, time)
    return rve.Observation(
        antpos, vis, 1 / sigma / sigma, pol, freq, auxiliary_tables={"ANTENNA": anttbl}
    )
