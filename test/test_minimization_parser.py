import pytest

from jubik.minimization_parser import (
    get_config_value, get_range_index, _delta_logic, n_samples_factory,
    sample_mode_factory, linear_sample_kwargs_factory,
    nonlinearly_update_kwargs_factory, kl_kwargs_factory, MinimizationParser
)

# Sample configuration
config = {
    'delta': {
        'switches': [0, 5],
        'values': [1.e-5, 1.e-6]
    },
    'samples': {
        'switches': [0, 10],
        'n_samples': [4, 5],
        'mode': ['nonlinear_resample', 'nonlinear_update'],
        'lin_maxiter': [60, 70],
        'nonlin_maxiter': [25, 35]
    },
    'kl_minimization': {
        'switches': [0, 10],
        'kl_maxiter': [10, 13]
    },
    'n_total_iterations': 15,
    'constants': {
        'switches': [0, 10],
        'domain_keys': [['diffuse_offset_mean', 'diffuse_fluctuations'], None]
    }
}


class TestMinimizationParser:

    @pytest.mark.parametrize("key, config, iteration, expected", [
        ('key1', {'key1': [1, 2, 3]}, 0, 1),
        ('key1', {'key1': [1, 2, 3]}, 3, 3),
        ('key2', {'key1': [1, 2, 3]}, 0, 0)
    ])
    def test_get_config_value(self, key, config, iteration, expected):
        assert get_config_value(key, config, iteration, 0) == expected

    @pytest.mark.parametrize("mini_cfg, iteration, total_iterations, expected",
                             [
                                 ({'switches': [0, 5, 10]}, 3, 10, 0),
                                 ({'switches': [0, 5, 10]}, 6, 10, 1),
                                 ({'switches': [0, 5, 10]}, 11, 15, 2)
                             ])
    def test_get_range_index(self, mini_cfg, iteration, total_iterations,
                             expected):
        assert get_range_index(mini_cfg, iteration,
                               total_iterations) == expected

    def test_get_range_index_missing_switches_defaults_to_zero(self):
        assert get_range_index({}, 0, 5) == 0

    def test_get_range_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            get_range_index({'switches': [0, 5]}, 10, 8)

    @pytest.mark.parametrize("type, config, switches_index, expected", [
        ('kl', {'values': [0.1, 0.2, 0.3]}, 0, 1.0),
        ('lin', {'values': [0.1, 0.2, 0.3]}, 0, 0.1),
        ('lin', {'values': [0.1, 0.2, 0.3]}, 1, 0.2),
        ('nonlin', {'values': [0.1, 0.2, 0.3]}, 0, 0.1)
    ])
    def test_delta_logic(self, type, config, switches_index, expected):
        assert _delta_logic(type, config, None,
                            0, switches_index, 10) == expected

    def test_delta_logic_passthrough_when_config_value_given(self):
        assert _delta_logic('lin', {'values': [0.1]}, 7.5, 0, 0, 10) == 7.5

    def test_delta_logic_unknown_keyword_raises(self):
        with pytest.raises(ValueError, match="Unknown keyword"):
            _delta_logic('invalid', {'values': [0.1]}, None, 0, 0, 10)

    def test_delta_logic_missing_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            _delta_logic('lin', None, None, 0, 0, 10)

    def test_delta_logic_missing_delta_value_raises(self):
        with pytest.raises(ValueError, match="delta value"):
            _delta_logic('lin', {'values': [None]}, None, 0, 0, 10)

    def test_n_samples_factory(self):
        n_samples = n_samples_factory(config)
        assert n_samples(0) == 4
        assert n_samples(11) == 5

    def test_n_samples_factory_raises_if_not_defined(self):
        bad_cfg = {
            'samples': {'switches': [0], 'mode': ['linear_sample']},
            'n_total_iterations': 2
        }
        with pytest.raises(ValueError, match="Number of samples"):
            n_samples_factory(bad_cfg)

    def test_sample_mode_factory(self):
        sample_mode = sample_mode_factory(config)
        assert sample_mode(0) == 'nonlinear_resample'
        assert sample_mode(11) == 'nonlinear_update'

    def test_sample_mode_factory_invalid_mode_raises(self):
        bad_cfg = {
            'samples': {'switches': [0], 'mode': ['not_a_mode']},
            'n_total_iterations': 1
        }
        with pytest.raises(ValueError, match="Unknown sample type"):
            sample_mode_factory(bad_cfg)

    def test_linear_sample_kwargs_factory(self):
        lin_kwargs = linear_sample_kwargs_factory(config, config['delta'],
                                                  10)
        kwargs = lin_kwargs(0)
        assert kwargs['cg_kwargs']['absdelta'] == 1.e-5
        assert kwargs['cg_kwargs']['maxiter'] == 60

    def test_nonlinearly_update_kwargs_factory(self):
        nonlin_kwargs = nonlinearly_update_kwargs_factory(config,
                                                          config['delta'])
        kwargs = nonlin_kwargs(11)
        assert kwargs['minimize_kwargs']['name'] == 'nonlin sampling'
        assert kwargs['minimize_kwargs']['xtol'] == 1.e-6
        assert kwargs['minimize_kwargs']['maxiter'] == 35

    def test_kl_kwargs_factory(self):
        kl_kwargs = kl_kwargs_factory(config, config['delta'], 10)
        kwargs = kl_kwargs(0)
        assert kwargs['minimize_kwargs']['name'] == 'kl'
        assert kwargs['minimize_kwargs']['absdelta'] == 1.e-4
        assert kwargs['minimize_kwargs']['maxiter'] == 10

    def test_minimization_parser(self):
        parser = MinimizationParser(config, n_dof=10)
        assert parser.n_samples(0) == 4
        assert parser.sample_mode(0) == 'nonlinear_resample'
        assert parser.draw_linear_kwargs(0)['cg_kwargs']['maxiter'] == 60
        assert parser.nonlinearly_update_kwargs(7)['minimize_kwargs'][
                   'xtol'] == 1.e-6
        assert parser.kl_kwargs(0)['minimize_kwargs']['maxiter'] == 10

    def test_minimization_parser_requires_n_dof_when_delta_is_present(self):
        with pytest.raises(ValueError, match="degrees of freedom"):
            MinimizationParser(config, n_dof=None)
