import pytest
import numpy as np

import jubik0 as ju
import nifty8.re as jft

@pytest.fixture
def sample_grid_info():
    return {
        'energy_bin': {
            'e_min': [0.5, 1.0],
            'e_max': [1.0, 2.0],
            'e_ref': [0.75, 1.5]
        },
        'sdim': 64,
        'edim': 4,
        's_padding_ratio': 0.1,
        'e_padding_ratio': 0.1
    }

@pytest.fixture
def sample_file_info():
    return {
        'res_dir': 'mock_test/',
        'data_dict': 'test_mock_data.pkl',
        'pos_dict': 'test_mock_pos.pkl'
    }

@pytest.fixture
def sample_tel_info():
    return {
        'fov': 30.0
    }

@pytest.fixture
def sample_prior_info():
    return {
        'diffuse': {
            'spatial': {
                'offset': {
                    'offset_mean': -14.,
                    'offset_std': [0.5, 0.05]},
                'fluctuations': {
                    'fluctuations': [0.5, 0.2],
                    'loglogavgslope': [-4.0, 0.3],
                    'flexibility': [0.4, 0.1],
                    'asperity': None,
                    'non_parametric_kind': 'power',
                    'harmonic_type': 'Fourier'
                },   
                'prefix': 'diffuse_spatial_'},
            'plaw': {
                'offset': {
                    'offset_mean': -14.,
                    'offset_std': [0.5, 0.05]},
                'fluctuations': {
                    'fluctuations': [0.5, 0.2],
                    'loglogavgslope': [-4.0, 0.3],
                    'flexibility': [0.4, 0.1],
                    'asperity': None,
                    'non_parametric_kind': 'power',
                    'harmonic_type': 'Fourier'
                },   
                'prefix': 'diffuse_spatial_'}}
    }

@pytest.fixture
def sample_response_dict():
    return {
        'R': lambda x: jft.Vector({1:x}),
        'mask': lambda x: jft.Vector({1:x}),
    }

@pytest.fixture
def sample_plot_info():
    return {
        'enabled': False,
    }
def test_create_mock_data(sample_grid_info, sample_file_info, sample_tel_info,
                          sample_prior_info, sample_plot_info,
                          sample_response_dict):
    seed = 42
    masked_mock_data = ju.create_mock_data(
        file_info=sample_file_info,
        grid_info=sample_grid_info,
        prior_info=sample_prior_info,
        tel_info=sample_tel_info,
        plot_info=sample_plot_info,
        seed=seed,
        response_dict=sample_response_dict
    )

    assert masked_mock_data is not None
    assert isinstance(masked_mock_data, jft.Vector)
    assert len(masked_mock_data.tree) > 0