# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from jax import vmap
from jax.tree_util import tree_map
from jax import numpy as jnp
import numpy as np

from jubik0.plot import _smooth


def calculate_uwr(pos, op, ground_truth, response_dict,
                   abs=False, exposure_mask=True, log=True):
    """
    Calculate the uncertainty-weighted residuals.
    
    The formula used is:
        (mean(op(pos)) - ground_truth) / std(op(pos))

    if ground_truth is None, it is set to 0.

    Parameters
    ----------
    pos : jft.Vector
        List of samples in latent space.
    op : jft.Model
        Operator for which the NWRs should be calculated, e.g. sky, point source, etc.
    response_dict : dict
        Dictionary containing the response information.
    abs : bool, optional
        If True, the absolute value of the residuals is returned. Default is False.
    exposure_mask : bool, optional
        If True, the exposure mask is applied. Default is True.
    log : bool, optional
        If True, the residuals are calculated in log space. Default is True.
    
    Returns
    -------
    res : jnp.ndarray   
        The uncertainty-weighted residuals.
    exposure_mask : jnp.ndarray
        The exposure mask.
        """
    op = vmap(op)
    if log:
        ground_truth = jnp.log(ground_truth) if ground_truth is not None else 0.
        res = (jnp.mean(jnp.log(op(pos)), axis=0) - ground_truth) / jnp.std(jnp.log(op(pos)),
                                                                            axis=0, ddof=1)
    else:
        ground_truth = 0. if ground_truth is None else ground_truth
        res = (jnp.mean(op(pos), axis=0) - ground_truth) / jnp.std(op(pos), axis=0, ddof=1)
    if abs:
        res = jnp.abs(res)
    if exposure_mask:
        exposure = response_dict['exposure']
        exposure_mask = jnp.array(exposure(jnp.ones_like(res))).sum(axis=0) > 0
    return res, exposure_mask


def calculate_nwr(pos, op, data, response_dict,
                   abs=False, min_counts=None, exposure_mask=True, response=True):
    """
    Calculate the noise-weighted residuals.

    The formula used is:
        (response(op(pos)) - data) / sqrt(response(op(pos)))
    
    Parameters
    ----------
    pos : jft.Vector
        List of samples in latent space.
    op : jft.Model
        Operator for which the NWRs should be calculated, e.g. sky, point source, etc.
    data : jnp.ndarray
        The data.
    response_dict : dict
        Dictionary containing the response information.
    abs : bool, optional
        If True, the absolute value of the residuals is returned. Default is False.
    min_counts : int, optional
        Minimum number of counts. Default is None.
    exposure_mask : bool, optional
        If True, the exposure mask is applied. Default is True.
    response : bool, optional
        If True, the response is applied. Default is True.

    Returns
    ------- 
    res : jnp.ndarray
        The noise-weighted residuals.
    tot_mask : jnp.ndarray
        The total mask.
    """
    if response:
        R = response_dict['R']
        kernel = response_dict['kernel']
    else:
        R = lambda x, k: x

    adj_mask = response_dict['mask_adj']
    sqrt = lambda x: tree_map(jnp.sqrt, x)
    res = lambda x: (R(op(x), kernel) - data) / sqrt(R(op(x), kernel))
    res = vmap(res, out_axes=1)
    res = jnp.array(vmap(adj_mask, in_axes=1, out_axes=1)(res(pos))[0])
    if abs:
        res = jnp.abs(res)

    min_count_mask = None
    if min_counts is not None:
        masked_indices = lambda x: jnp.array(x < min_counts, dtype=float)
        masked_indices = tree_map(masked_indices, data)
        min_count_mask = lambda x: adj_mask(masked_indices)[0]
    if exposure_mask:
        exp_mask = lambda x: response_dict['exposure'](jnp.ones(op(x).shape)) == 0.
        if min_count_mask is not None:
            tot_mask = lambda x: np.logical_or(min_count_mask(x), exp_mask(x), dtype=bool)
        else:
            tot_mask = exp_mask
    else:
        tot_mask = min_count_mask if min_count_mask is not None else None
    if tot_mask is not None:
        tot_mask = vmap(tot_mask, out_axes=1)
    return res, tot_mask(pos)


def calculate_convolved_nwr(
    samples, 
    op, 
    data, 
    response_dict,
    sigma=1.,
    abs=True, 
    min_counts=None, 
    exposure_mask=True, 
    response=True
):
    """
    Calculate the noise-weighted residuals.

    The formula used is:
        gauss * |response(op(pos)) - data| / sqrt(gauss * response(op(pos)))
    
    Parameters
    ----------
    samples : jft.Samples
        List of samples in latent space.
    op : jft.Model
        Operator for which the NWRs should be calculated, e.g. sky, point source, etc.
    data : jnp.ndarray
        The data.
    response_dict : dict
        Dictionary containing the response information.
    sigma : float, optional
        The smoothing parameter. Default is 1.
    abs : bool, optional
        If True, the absolute value of the residuals is returned. Default is False.
    min_counts : int, optional
        Minimum number of counts. Default is None.
    exposure_mask : bool, optional
        If True, the exposure mask is applied. Default is True.
    response : bool, optional
        If True, the response is applied. Default is True.

    Returns
    ------- 
    res : jnp.ndarray
        The noise-weighted residuals.
    tot_mask : jnp.ndarray
        The total mask.
    """
    if response:
        R = response_dict['R']
        kernel = response_dict['kernel']
    else:
        R = lambda x, k: x

    adj_mask = lambda x : response_dict['mask_adj'](x)[0]
    absolute = lambda x: tree_map(jnp.abs, x)
    conv = lambda x: _smooth(sigma, x)
    convolution = vmap(conv, in_axes=1, out_axes=1)

    # norm = 2. * np.sqrt(np.pi*sigma**2)
    norm = 1.

    if abs:
        res = lambda x: norm * convolution(adj_mask(absolute((R(op(x), kernel) - data)))) / convolution(jnp.sqrt(adj_mask(R(op(x), kernel))))
    else:
        res = lambda x: norm * convolution(adj_mask((R(op(x), kernel) - data))) / convolution(jnp.sqrt(adj_mask(R(op(x), kernel))))
    
    conv_res = tuple(res(s) for s in samples)
    conv_res = np.array(conv_res)

    min_count_mask = None
    if min_counts is not None:
        masked_indices = lambda x: jnp.array(x < min_counts, dtype=float)
        masked_indices = tree_map(masked_indices, data)
        min_count_mask = lambda x: adj_mask(masked_indices)[0]
    if exposure_mask:
        exp_mask = lambda x: response_dict['exposure'](jnp.ones(op(x).shape)) == 0.
        if min_count_mask is not None:
            tot_mask = lambda x: np.logical_or(min_count_mask(x), exp_mask(x), dtype=bool)
        else:
            tot_mask = exp_mask
    else:
        tot_mask = min_count_mask if min_count_mask is not None else None
    # if tot_mask is not None:
        # tot_mask = vmap(tot_mask, out_axes=1)
    return conv_res, tot_mask(samples.pos)