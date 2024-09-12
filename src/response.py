# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

import nifty8.re as jft
import numpy as np


def build_exposure_function(exposures, exposure_cut=None):
    """
    Returns a function that applies instrument exposures to an input array.

    Parameters
    ----------
    exposures : ndarray
        Array with instrument exposure maps. The 0-th axis indexes the telescope
         module (for multi-module instruments).
    exposure_cut : float or None, optional
        A threshold exposure value below which exposures are set to zero.
        If None (default), no threshold is applied.

    Returns
    -------
    callable
        A function that takes an input array `x` and returns the element-wise
        product of `exposures` and `x`, with the first dimension of `exposures`
        broadcasted to match the shape of `x`.

    Raises
    ------
    ValueError:
        If `exposures` is not a 2D array or `exposure_cut` is negative.
    """
    if exposure_cut is not None:
        if exposure_cut < 0:
            raise ValueError("exposure_cut should be non-negative or None!")
        exposures[exposures < exposure_cut] = 0
    # FIXME short hack to remove additional axis. Also the Ifs should be
    #  restructed

    def exposure(x): return exposures * x
    return exposure


def build_readout_function(flags, threshold=None, keys=None):
    """
    Applies a readout corresponding to input flags.

    Parameters
    ----------
        flags : ndarray
        Array with flags. Where flags are equal to zero the input will not be
        read out.
        The 0-th axis indexes the number of 3D flag maps, e.g. it could index
        the telescope module
        (for multi-module instruments exposure maps).
        The 1st axis indexes the energy direction.
        The 2nd and 3rd axis refer to the spatial direction.
        threshold: float or None, optional
            A threshold value below which flags are set to zero (e.g.,
            an exposure cut).
            If None (default), no threshold is applied.
        keys : list or tuple or None
            A list or tuple containing the keys for the response output
            dictionary.
            For example, a list of the telescope modules ids for a
            multi-module instrument.
            Optional for a single-module observation.
    Returns
    -------
        function: A callable that applies a mask to an input array (e.g. an
        input sky) and returns
        a `nifty8.re.Vector` containing a dictionary of read-out inputs.
    Raises:
    -------
        ValueError:
        If threshold is negative.
        If keys does not have the right shape.
        If the flags do not have the right shape.
    """
    if threshold < 0:
        raise ValueError("threshold should be positive!")
    if threshold is not None:
        flags[flags < threshold] = 0
    mask = flags == 0

    if keys is None:
        keys = ['masked input']
    elif len(keys) != flags.shape[0]:
        raise ValueError("length of keys should match the number of flag maps.")

    def _apply_readout(x: np.array):
        """
        Reads out input array (e.g, sky signals to which an exposure is applied)
        at locations specified by a mask.
        Args:
            x: ndarray

        Returns:
            readout = `nifty8.re.Vector` containing a dictionary of read-out
            inputs.
        """
        if len(mask.shape) != 4:
            raise ValueError("flags should have shape (n, m, q, l)!")

        if len(x.shape) != 4:
            raise ValueError("input should have shape (n, m, q, l)!")
        return jft.Vector({key: x[i][~mask[i]] for i, key in enumerate(keys)})

    return _apply_readout
