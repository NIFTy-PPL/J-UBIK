import nifty.re as jft
import pytest
from jax import numpy as jnp
from jax import config, random

import jubik as ju

config.update('jax_enable_x64', True)


@pytest.fixture
def priors():
    priors_dict = {}
    priors_dict['diffuse'] = {}
    priors_dict['diffuse']['spatial'] = {}
    priors_dict['diffuse']['spatial']['offset'] = {}
    priors_dict['diffuse']['spatial']['offset']['offset_mean'] = -14.0
    priors_dict['diffuse']['spatial']['offset']['offset_std'] = [0.5, 0.05]

    priors_dict['diffuse']['spatial']['fluctuations'] = {}
    priors_dict['diffuse']['spatial']['fluctuations']['fluctuations'] = [0.5,
                                                                       0.2]
    priors_dict['diffuse']['spatial']['fluctuations']['loglogavgslope'] =\
    [-4.0, 0.3]
    priors_dict['diffuse']['spatial']['fluctuations']['flexibility'] = \
    [0.4, 0.1]
    priors_dict['diffuse']['spatial']['fluctuations']['asperity'] = None
    priors_dict['diffuse']['spatial']['fluctuations']\
        ['non_parametric_kind'] = 'power'
    priors_dict['diffuse']['spatial']['prefix'] = 'diffuse_spatial_'

    priors_dict['diffuse']['plaw'] = {}
    priors_dict['diffuse']['plaw']['offset'] = {}
    priors_dict['diffuse']['plaw']['offset']['offset_mean'] = -2.0
    priors_dict['diffuse']['plaw']['offset']['offset_std'] = [0.3, 0.05]

    priors_dict['diffuse']['plaw']['fluctuations'] = {}
    priors_dict['diffuse']['plaw']['fluctuations']['fluctuations'] = [0.5,
                                                                       0.2]
    priors_dict['diffuse']['plaw']['fluctuations']['loglogavgslope'] =\
    [-4.0, 0.3]
    priors_dict['diffuse']['plaw']['fluctuations']['flexibility'] = \
    [0.4, 0.1]
    priors_dict['diffuse']['plaw']['fluctuations']['asperity'] = None
    priors_dict['diffuse']['plaw']['fluctuations']\
        ['non_parametric_kind'] = 'power'
    priors_dict['diffuse']['plaw']['prefix'] = 'diffuse_plaw_'

    return priors_dict

@pytest.fixture
def config(priors):
    cfg_dict = {}
    cfg_dict['sdim'] = 128
    cfg_dict['edim'] = 3
    cfg_dict['s_padding_ratio'] = 1.1
    cfg_dict['e_padding_ratio'] = 1.0
    cfg_dict['fov'] = 4096
    cfg_dict['e_min'] = [0.2, 1.0, 2.0]
    cfg_dict['e_max'] = [1.0, 2.0, 4.0]
    cfg_dict['e_ref'] = 2.0
    cfg_dict['priors'] = priors
    return cfg_dict

@pytest.fixture
def sky_model():
    return ju.SkyModel()

def test_sky_model_creation(sky_model, config):
    assert sky_model is not None
    sky = sky_model.create_sky_model(**config)
    assert isinstance(sky, jft.Model)
    sky_dict = sky_model.sky_model_to_dict()
    assert 'sky' in sky_dict

def test_sky_application(sky_model, config):
    sky = sky_model.create_sky_model(**config)
    key = random.PRNGKey(81)
    key, subkey = random.split(key)

    pos = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    sky_real = sky(pos)
    assert isinstance(sky_real, jnp.ndarray)

