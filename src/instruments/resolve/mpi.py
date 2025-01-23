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
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import functools

import nifty8 as ift

from .util import my_asserteq
from .logger import logger

try:
    from mpi4py import MPI

    master = MPI.COMM_WORLD.Get_rank() == 0
    comm = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF
    ntask = comm.Get_size()
    rank = comm.Get_rank()
    master = rank == 0
    mpi = ntask > 1

    if ntask == 1:
        master = True
        mpi = False
        comm = None
        comm_self = None
        rank = 0
except ImportError:
    logger.warning("Could not import MPI")
    master = True
    mpi = False
    comm = None
    comm_self = None
    rank = 0


def onlymaster(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not master:
            return
        state0 = ift.random.getState()
        f = func(*args, **kwargs)
        state1 = ift.random.getState()
        my_asserteq(state0, state1)
        return f

    return wrapper


def barrier(comm=None):
    if comm is None:
        return
    comm.Barrier()
