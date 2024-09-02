from jax import vmap
from jax.tree_util import tree_map
from jax import numpy as jnp


def calculate_uwr(pos, op, ground_truth, response_dict,
                   abs=False, exposure_mask=True, log=True):
    op = jax.vmap(op)
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
        exposure_mask = np.array(exposure(np.ones_like(res))).sum(axis=0) > 0
    return res, exposure_mask


def calculate_nwr(pos, op, data, response_dict,
                   abs=False, min_counts=None, exposure_mask=True, response=True):
    if response:
        R = response_dict['R']
    else:
        R = lambda x: x

    adj_mask = response_dict['mask_adj']
    sqrt = lambda x: tree_map(jnp.sqrt, x)
    res = lambda x: (R(op(x)) - data) / sqrt(R(op(x)))
    res = jax.vmap(res, out_axes=1)
    res = np.array(jax.vmap(adj_mask, in_axes=1, out_axes=1)(res(pos))[0])
    if abs:
        res = np.abs(res)

    min_count_mask = None
    if min_counts is not None:
        masked_indices = lambda x: np.array(x < min_counts, dtype=float)
        masked_indices = tree_map(masked_indices, data)
        min_count_mask = lambda x: adj_mask(masked_indices)[0]
    if exposure_mask:
        exp_mask = lambda x: response_dict['exposure'](np.ones(op(x).shape)) == 0.
        if min_count_mask is not None:
            tot_mask = lambda x: np.logical_or(min_count_mask(x), exp_mask(x), dtype=bool)
        else:
            tot_mask = exp_mask
    else:
        tot_mask = min_count_mask if min_count_mask is not None else None
    if tot_mask is not None:
        tot_mask = jax.vmap(tot_mask, out_axes=1)
    return res, tot_mask(pos)

