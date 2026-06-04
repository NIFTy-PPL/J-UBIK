from pathlib import Path

import nifty.re as jft
import numpy as np
import pytest

import jubik as ju


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
def sample_file_info(tmp_path):
    return {
        'res_dir': str(tmp_path / 'mock_test'),
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


def _assert_dict_arrays_equal(lhs, rhs):
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        np.testing.assert_array_equal(np.asarray(lhs[key]), np.asarray(rhs[key]))


def _assert_vector_tree_equal(lhs, rhs):
    _assert_dict_arrays_equal(lhs.tree, rhs.tree)


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
    assert set(masked_mock_data.tree.keys()) == {1}

    for arr in masked_mock_data.tree.values():
        arr_np = np.asarray(arr)
        assert np.issubdtype(arr_np.dtype, np.integer)
        assert np.isfinite(arr_np).all()
        assert (arr_np >= 0).all()
        assert arr_np.size > 0

    data_path = Path(sample_file_info['res_dir']) / sample_file_info['data_dict']
    pos_path = Path(sample_file_info['res_dir']) / sample_file_info['pos_dict']
    assert data_path.exists()
    assert pos_path.exists()

    saved_data = ju.load_from_pickle(data_path)
    saved_pos = ju.load_from_pickle(pos_path)

    _assert_dict_arrays_equal(saved_data, masked_mock_data.tree)
    assert len(saved_pos) > 0
    for arr in saved_pos.values():
        arr_np = np.asarray(arr)
        assert np.isfinite(arr_np).all()
        assert arr_np.size > 0


def test_create_mock_data_seed_reproducibility(sample_grid_info, sample_file_info,
                                               sample_tel_info, sample_prior_info,
                                               sample_plot_info, sample_response_dict):
    def run_once(seed, subdir_name):
        file_info = dict(sample_file_info)
        base_dir = Path(sample_file_info['res_dir']).parent
        file_info['res_dir'] = str(base_dir / subdir_name)
        data = ju.create_mock_data(
            file_info=file_info,
            grid_info=sample_grid_info,
            prior_info=sample_prior_info,
            tel_info=sample_tel_info,
            plot_info=sample_plot_info,
            seed=seed,
            response_dict=sample_response_dict
        )
        pos = ju.load_from_pickle(Path(file_info['res_dir']) / file_info['pos_dict'])
        return data, pos

    data_a, pos_a = run_once(seed=42, subdir_name='mock_test_seed_a')
    data_b, pos_b = run_once(seed=42, subdir_name='mock_test_seed_b')
    data_c, pos_c = run_once(seed=43, subdir_name='mock_test_seed_c')

    _assert_vector_tree_equal(data_a, data_b)
    _assert_dict_arrays_equal(pos_a, pos_b)

    assert pos_a.keys() == pos_c.keys()
    assert any(
        not np.array_equal(np.asarray(pos_a[key]), np.asarray(pos_c[key]))
        for key in pos_a
    )
