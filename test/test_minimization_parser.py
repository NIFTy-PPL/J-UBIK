from jubik0.library.minimization_parser import get_config_value, get_range_index, _delta_logic, \
    n_samples_factory, sample_mode_factory, linear_sample_kwargs_factory, \
    nonlinearly_update_kwargs_factory, kl_kwargs_factory, MinimizationParser

# Sample configuration
config = {
    'delta': {
        'switches': [0, 10],
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
    'n_total_iterations': 15
}


class TestMinimizationParser:
    def test_get_config_value(self):
        config = {'key1': [1, 2, 3]}
        assert get_config_value('key1', config, 0, 0) == 1
        assert get_config_value('key1', config, 3, 0) == 3
        assert get_config_value('key2', config, 0, 0) == 0

    def test_get_range_index(self):
        mini_cfg = {'switches': [0, 5, 10]}
        assert get_range_index(mini_cfg, 3, 10) == 0
        assert get_range_index(mini_cfg, 6, 10) == 1
        assert get_range_index(mini_cfg, 11, 15) == 2

    def test_delta_logic_kl(self):
        delta_config = {'values': [0.1, 0.2, 0.3]}
        assert _delta_logic('kl', delta_config, None, 0,
                            0, 10) == 1.0

    def test_delta_logic_linear(self):
        delta_config = {'values': [0.1, 0.2, 0.3]}
        assert _delta_logic('linear', delta_config, None, 0,
                            0, 10) == 0.1
        assert _delta_logic('linear', delta_config, None, 11,
                            1, 10) == 0.2

    def test_delta_logic_nonlinear(self):
        delta_config = {'values': [0.1, 0.2, 0.3]}
        assert _delta_logic('nonlinear', delta_config, None, 0, 0, None) == 0.1

    def test_n_samples_factory(self):
        n_samples = n_samples_factory(config)
        assert n_samples(0) == 4
        assert n_samples(11) == 5

    def test_sample_mode_factory(self):
        sample_mode = sample_mode_factory(config)
        assert sample_mode(0) == 'nonlinear_resample'
        assert sample_mode(11) == 'nonlinear_update'

    def test_linear_sample_kwargs_factory(self):
        lin_kwargs = linear_sample_kwargs_factory(config, config['delta'], 10)
        kwargs = lin_kwargs(0)
        assert kwargs['cg_kwargs']['absdelta'] == 1.e-5
        assert kwargs['cg_kwargs']['maxiter'] == 60

    def test_nonlinearly_update_kwargs_factory(self):
        nonlin_kwargs = nonlinearly_update_kwargs_factory(config, config['delta'])
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
        assert parser.nonlinearly_update_kwargs(11)['minimize_kwargs']['maxiter'] == 35
        assert parser.kl_kwargs(0)['minimize_kwargs']['maxiter'] == 10