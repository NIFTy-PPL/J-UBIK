import pytest
import numpy as np
from jax import numpy as jnp
from jax import random

import jubik0 as ju
import nifty8.re as jft

@pytest.fixture
def config():
    return 'config_test_sky_model.yaml'

@pytest.fixture
def sky_model(config):
    return ju.SkyModel(config)

def test_sky_model_creation(sky_model):
    assert sky_model is not None
    sky = sky_model.create_sky_model()
    assert isinstance(sky, jft.Model)
    sky_dict = sky_model.sky_model_to_dict()
    assert 'sky' in sky_dict

def test_sky_application(sky_model):
    sky = sky_model.create_sky_model()
    key = random.PRNGKey(81)
    key, subkey = random.split(key)

    pos = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    sky_real = sky(pos)
    assert isinstance(sky_real, jnp.ndarray)
