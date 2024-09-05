# test_diagnostics.py

import pytest
import jax.numpy as jnp
from jax import random

import jubik0 as ju


@pytest.fixture
def sample_data():
    config = 'config_test'
    seed = 42
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    sky = jnp.exp(random.normal(subkey, (100, 100)))
    data = random.poisson(subkey, sky).astype(int)
    op = lambda x: x
    response_dict = {
        'exposure': lambda x: x,
        'R': lambda x: x,
        'mask_adj': lambda x: x
    }
    return sky, op, sky, data, response_dict


def test_calculate_uwr(sample_data):
    pos, op, ground_truth, _, response_dict = sample_data
    res, exposure_mask = ju.calculate_uwr(pos, op, ground_truth, response_dict,
                                          abs=True, exposure_mask=True, log=True)

    assert res is not None
    assert exposure_mask is not None
    assert res.shape == ground_truth.shape


def test_calculate_nwr(sample_data):
    pos, op, _, data, response_dict = sample_data
    res, tot_mask = ju.calculate_nwr(pos, op, data, response_dict, abs=True,
                                     min_counts=1, exposure_mask=True, response=True)

    assert res is not None
    assert tot_mask is not None
    assert res.shape == data.shape