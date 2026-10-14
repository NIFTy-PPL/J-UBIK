import pytest

from jubik.minimization_parser import (
    get_config_value, get_range_index, _delta_logic, n_samples_factory,
    sample_mode_factory, linear_sample_kwargs_factory,
    nonlinearly_update_kwargs_factory, kl_kwargs_factory, MinimizationParser,
    constants_factory, point_estimates_factory
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
    },
    'point_estimates': {
        'switches': [0, 10],
        'domain_keys': [['point_estimate_a'], None]
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

    def test_get_range_index_after_total_uses_last_range(self):
        assert get_range_index({'switches': [0, 5]}, 10, 8) == 2

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

    def test_delta_logic_missing_delta_inherits_default(self):
        assert _delta_logic('lin', None, None, 0, 0, 10) is None

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

    def test_kl_cg_kwargs_are_read_from_kl_minimization(self):
        cfg = {
            **config,
            'samples': {**config['samples'], 'switches': [0, 5],
                        'kl_cg_maxiter': [99, 99],
                        'kl_cg_miniter': [99, 99],
                        'kl_cg_absdelta': [99, 99],
                        'kl_cg_atol': [99, 99]},
            'kl_minimization': {**config['kl_minimization'],
                                'kl_cg_maxiter': [7, 8],
                                'kl_cg_miniter': [2, 3],
                                'kl_cg_absdelta': [0.2, 0.02],
                                'kl_cg_atol': [0.1, 0.01]},
        }
        kl_kwargs = kl_kwargs_factory(cfg, cfg['delta'], 10)
        assert kl_kwargs(0)['minimize_kwargs']['cg_kwargs'] == {
            'name': 'kl_cg', 'absdelta': 0.2, 'atol': 0.1,
            'miniter': 2, 'maxiter': 7,
        }
        assert kl_kwargs(11)['minimize_kwargs']['cg_kwargs'] == {
            'name': 'kl_cg', 'absdelta': 0.02, 'atol': 0.01,
            'miniter': 3, 'maxiter': 8,
        }

    def test_minimization_parser(self):
        parser = MinimizationParser(config, n_dof=10)
        assert parser.n_samples(0) == 4
        assert parser.sample_mode(0) == 'nonlinear_resample'
        assert parser.draw_linear_kwargs(0)['cg_kwargs']['maxiter'] == 60
        assert parser.nonlinearly_update_kwargs(7)['minimize_kwargs'][
                   'xtol'] == 1.e-6
        assert parser.kl_kwargs(0)['minimize_kwargs']['maxiter'] == 10
        assert parser.constants(0) == (
            'diffuse_offset_mean', 'diffuse_fluctuations'
        )
        assert parser.constants(11) is None
        assert parser.point_estimates(0) == ('point_estimate_a',)
        assert parser.point_estimates(11) is None

    def test_minimization_parser_requires_n_dof_when_delta_is_present(self):
        with pytest.raises(ValueError, match="degrees of freedom"):
            MinimizationParser(config, n_dof=None)


@pytest.mark.parametrize("delta", [None, "absent"])
def test_unspecified_solver_options_inherit_nifty_defaults(delta):
    cfg = {'samples': {'n_samples': 1, 'mode': 'nonlinear_resample'},
           'kl_minimization': {}, 'n_total_iterations': 2}
    if delta is None:
        cfg['delta'] = None
    parser = MinimizationParser(cfg, verbose=False)
    assert parser.draw_linear_kwargs(0) == {
        'cg_name': None, 'cg_kwargs': {'name': 'lin sampling_cg'}}
    assert parser.nonlinearly_update_kwargs(0) == {
        'minimize_kwargs': {'name': 'nonlin sampling',
                            'cg_kwargs': {'name': 'nonlin_cg'}}}
    assert parser.kl_kwargs(0) == {
        'minimize_kwargs': {'name': 'kl', 'cg_kwargs': {'name': 'kl_cg'}}}
    assert parser.constants(0) is None
    assert parser.point_estimates(0) is None


@pytest.mark.parametrize("prefix,option", [
    ('lin', 'resnorm'), ('lin', 'norm_ord'),
    ('nonlin', 'absdelta'), ('nonlin', 'norm_ord'),
    ('nonlin', 'energy_reduction_factor'),
    ('nonlin_cg', 'resnorm'), ('nonlin_cg', 'norm_ord'),
    ('kl', 'norm_ord'), ('kl', 'energy_reduction_factor'),
    ('kl_cg', 'tol'), ('kl_cg', 'resnorm'), ('kl_cg', 'norm_ord'),
])
def test_new_solver_options_follow_their_own_schedule(prefix, option):
    section = 'kl_minimization' if prefix.startswith('kl') else 'samples'
    cfg = {**config, section: {**config[section], 'switches': [0, 3],
                              f'{prefix}_{option}': [2, 3]}}
    parser = MinimizationParser(cfg, n_dof=10, verbose=False)

    def options(iteration):
        if prefix == 'lin':
            return parser.draw_linear_kwargs(iteration)['cg_kwargs']
        factory = parser.kl_kwargs if prefix.startswith('kl') else (
            parser.nonlinearly_update_kwargs)
        kwargs = factory(iteration)['minimize_kwargs']
        return kwargs['cg_kwargs'] if prefix.endswith('_cg') else kwargs

    assert options(0)[option] == 2
    assert options(2)[option] == 3
    assert options(14)[option] == 3


def test_null_inherits_but_scalar_zero_is_forwarded():
    cfg = {**config, 'delta': None,
           'samples': {**config['samples'], 'lin_atol': None,
                       'nonlin_xtol': [None],
                       'nonlin_cg_absdelta': [None]},
           'kl_minimization': {**config['kl_minimization'],
                               'kl_xtol': None, 'kl_cg_absdelta': [None],
                               'kl_energy_reduction_factor': 0,
                               'kl_cg_tol': 0}}
    parser = MinimizationParser(cfg, verbose=False)
    assert 'atol' not in parser.draw_linear_kwargs(0)['cg_kwargs']
    nonlin = parser.nonlinearly_update_kwargs(0)['minimize_kwargs']
    assert 'xtol' not in nonlin
    assert 'absdelta' not in nonlin['cg_kwargs']
    kl = parser.kl_kwargs(0)['minimize_kwargs']
    assert 'xtol' not in kl
    assert 'absdelta' not in kl['cg_kwargs']
    assert kl['energy_reduction_factor'] == 0
    assert kl['cg_kwargs']['tol'] == 0


def test_explicit_tolerances_override_delta_and_null_uses_delta():
    cfg = {**config,
           'samples': {**config['samples'], 'lin_absdelta': [None, 0],
                       'nonlin_xtol': [None, 0]},
           'kl_minimization': {**config['kl_minimization'],
                               'kl_absdelta': [None, 0]}}
    parser = MinimizationParser(cfg, n_dof=10, verbose=False)
    assert parser.draw_linear_kwargs(0)['cg_kwargs']['absdelta'] == 1e-5
    assert parser.nonlinearly_update_kwargs(0)['minimize_kwargs']['xtol'] == 1e-5
    assert parser.kl_kwargs(0)['minimize_kwargs']['absdelta'] == 1e-4
    assert parser.draw_linear_kwargs(11)['cg_kwargs']['absdelta'] == 0
    assert parser.nonlinearly_update_kwargs(11)['minimize_kwargs']['xtol'] == 0
    assert parser.kl_kwargs(11)['minimize_kwargs']['absdelta'] == 0


def test_nonlinear_verbose_false_does_not_log(monkeypatch):
    from jubik import minimization_parser
    messages = []
    monkeypatch.setattr(minimization_parser.jft.logger, 'info', messages.append)
    kwargs = nonlinearly_update_kwargs_factory(config, config['delta'],
                                               verbose=False)
    kwargs(0)
    assert messages == []


@pytest.mark.parametrize("switches", [[], [1], [0, 0], [0, 3, 2],
                                      [0, 6], [0, 1.5], [False], '0'])
def test_invalid_switches_fail_clearly(switches):
    with pytest.raises(ValueError, match='switches'):
        get_range_index({'switches': switches}, 0, 5)


def test_empty_value_schedule_fails_clearly():
    with pytest.raises(ValueError, match='lin_maxiter'):
        get_config_value('lin_maxiter', {'lin_maxiter': []}, 0, None)


@pytest.mark.parametrize("section,factory", [
    ('constants', constants_factory), ('point_estimates', point_estimates_factory)
])
def test_domain_key_schedules(section, factory):
    cfg = {section: {'switches': [0, 3], 'domain_keys': [['a', 'b'], None]},
           'n_total_iterations': 5}
    keys = factory(cfg, verbose=False)
    assert keys(0) == ('a', 'b')
    assert keys(2) is None
    assert keys(5) is None
    assert factory({'n_total_iterations': 1}, verbose=False)(0) is None
    cfg[section]['domain_keys'] = [[]]
    assert factory(cfg, verbose=False)(0) == ()
    cfg[section]['domain_keys'] = ['a', 'b']
    with pytest.raises(ValueError, match=f'{section}.domain_keys'):
        factory(cfg, verbose=False)


def test_kl_defaults_match_native_newton_cg():
    import jax.numpy as jnp
    import nifty.re as jft
    from numpy.testing import assert_allclose

    cfg = {**config, 'kl_minimization': {'kl_absdelta': 1e-4}}
    kwargs = kl_kwargs_factory(cfg, cfg['delta'], 10, verbose=False)(0)[
        'minimize_kwargs']
    diagonal = jnp.array([1., 5., 20.])

    def fun_and_grad(x):
        return 0.5 * jnp.vdot(x, diagonal * x), diagonal * x

    def solve(options):
        thresholds = []

        def cg(mat, rhs, **cg_kwargs):
            thresholds.append((float(cg_kwargs['absdelta']),
                               float(cg_kwargs['resnorm'])))
            return jft.conjugate_gradient._cg(mat, rhs, **cg_kwargs)

        result = jft.optimize._newton_cg(
            x0=jnp.array([3., 2., 1.]), fun_and_grad=fun_and_grad,
            hessp=lambda x, v: diagonal * v, cg=cg, **options)
        return result, thresholds

    parsed, thresholds = solve(kwargs)
    native, native_thresholds = solve({
        'name': 'kl', 'absdelta': 1e-4, 'cg_kwargs': {'name': 'kl_cg'}})
    assert thresholds[0][0] == pytest.approx(1e-6)
    assert thresholds[0][1] > 0
    assert_allclose(thresholds, native_thresholds)
    assert_allclose(parsed.x, native.x)
    assert parsed.nit == native.nit
    assert float(parsed.fun) < 1e-4
