# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import functools
import nifty8 as ift

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    master = True
    comm = None


def onlymaster(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not master:
            return
        state0 = ift.random.getState()
        f = func(*args, **kwargs)
        state1 = ift.random.getState()
        assert state0 == state1
        return f

    return wrapper
#This should be part of the script
