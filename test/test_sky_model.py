import nifty.re as jft
import numpy as np
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
    cfg_dict['shape'] = 128
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
    assert 'diffuse' in sky_dict
    assert sky_dict['sky'] is sky
    assert all(model is not None for model in sky_dict.values())

def test_sky_application(sky_model, config):
    sky = sky_model.create_sky_model(**config)
    key = random.PRNGKey(81)
    key, subkey = random.split(key)

    pos = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    sky_real = sky(pos)
    assert isinstance(sky_real, jnp.ndarray)
    assert sky_real.shape == sky.target.shape
    assert bool(jnp.all(jnp.isfinite(sky_real)))
    assert bool(jnp.all(sky_real >= 0))
    assert float(jnp.var(sky_real)) > 0

    sky_real_repeat = sky(pos)
    np.testing.assert_allclose(np.asarray(sky_real), np.asarray(sky_real_repeat))


def test_rectangular_public_geometry_becomes_internal_yx(sky_model, config):
    config = dict(config, shape=(16, 8), fov=(4096, 2048))
    sky = sky_model.create_sky_model(**config)
    assert sky.target.shape[-2:] == (8, 16)
    assert sky_model.s_distances == (256.0, 256.0)


def test_deprecated_sdim_argument_warns_and_builds(sky_model, config):
    config = dict(config, shape=None, sdim=16)
    with pytest.warns(FutureWarning, match="2026-12-17"):
        sky = sky_model.create_sky_model(**config)
    assert sky.target.shape[-2:] == (16, 16)


def test_deprecated_sdim_argument_clashes_with_shape(sky_model, config):
    with pytest.raises(ValueError, match="drop `sdim`"):
        sky_model.create_sky_model(**dict(config, shape=16, sdim=16))


def test_rectangular_sdim_argument_is_rejected(sky_model, config):
    with pytest.raises(ValueError, match="axis order is undefined"):
        sky_model.create_sky_model(**dict(config, shape=None, sdim=(16, 8)))
