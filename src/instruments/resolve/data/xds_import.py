# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2019-2021 Max-Planck-Society

import os
from os.path import expanduser, isdir

import numpy as np

from .antenna_positions import AntennaPositions
from .auxiliary_table import AuxiliaryTable
from .observation import Observation
from .polarization import Polarization


def xds2observations(xds):
    from daskms.experimental.zarr import xds_from_zarr
    
    # Input checks
    xds = expanduser(xds)
    if xds[-1] == "/":
        xds = xds[:-1]
    if not isdir(xds):
        raise RuntimeError
    if xds == ".":
        xds = os.getcwd()
    
    xds = xds_from_zarr(xds)
    observations = []
    for ii in range(len(xds)):
        uvw = xds[ii]["UVW"].values
        ant1 = ant2 = time = None
        vis = xds[ii]["VIS"].values
        vis = np.expand_dims(vis, 0)
        wgt = xds[ii]["WEIGHT"].values
        wgt = np.expand_dims(wgt, 0)
        freq = xds[ii]['FREQ'].values
        print(f"xds: {ii}")
        print(f"uvw: {uvw.shape}")
        print(f"vis: {vis.shape}")
        print(f"wgt: {wgt.shape}")
        print(f"freq: {freq.shape}")
        dir = np.zeros((1,1,2))
        dir[0,0,0] = xds[ii].ra
        dir[0,0,1] = xds[ii].dec
        field_table = AuxiliaryTable({'REFERENCE_DIR': dir})
        auxtables = {'FIELD': field_table}
        pol = xds[ii].product
        if pol == 'I':
            polobj = Polarization([])
        else:
            raise NotImplementedError('please implement!')
        
        antpos = AntennaPositions(uvw, ant1, ant2, time)
        obs = Observation(antpos, vis, wgt, polobj, freq,
                          auxiliary_tables=auxtables)
        observations.append(obs)
    return observations

def combine_observations(observations_list):
    freq = observations_list[0].freq
    polarization = observations_list[0].polarization
    for obs in observations_list:
        if not np.all(freq == obs.freq):
            raise ValueError('all obs need to have the same freq')
        if not polarization == obs.polarization:
            raise ValueError('all obs need to have the same polarization')
    new_vis = np.concatenate([obs._vis for obs in observations_list], axis=1)
    new_uvw = np.concatenate([obs.uvw for obs in observations_list])
    new_weight = np.concatenate([obs._weight for obs in observations_list], axis=1)
    cal_info = [obs.ant1 != None for obs in observations_list]
    if np.all(cal_info):
        new_ant1 = np.concatenate([obs.ant1 for obs in observations_list])
        new_ant2 = np.concatenate([obs.ant2 for obs in observations_list])
        new_time = np.concatenate([obs.time for obs in observations_list])
    else:
        new_ant1 = new_ant2 = new_time = None

    new_antenna_positions = AntennaPositions(new_uvw, new_ant1, new_ant2, new_time)
    return Observation(new_antenna_positions, new_vis, new_weight, polarization, freq)

