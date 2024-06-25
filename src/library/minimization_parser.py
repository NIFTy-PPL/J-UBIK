from typing import Callable, Union, Any, Optional
import nifty8.re as jft

import logging

logging.basicConfig(level=logging.INFO)

# CONFIGURATION CONSTANTS
SWITCHES = 'switches'
SAMPLES = 'samples'
N_SAMPLES = 'n_samples'
N_TOTAL_ITERATIONS = 'n_total_iterations'
MODE = 'mode'
DELTA = 'delta'
LINKEY = 'lin'
ABSDELTA = 'absdelta'
MINITER = 'miniter'
MAXITER = 'maxiter'
DELTA_VALUE = 'values'
ATOL = 'atol'
XTOL = 'xtol'
NONLIN = 'nonlin'
NONLIN_CG = 'nonlin_cg'
KL_MINI = 'kl_minimization'
KL = 'kl'


def get_config_value(
    key: str, config: dict, index: int, default: Any, verbose: bool = False
) -> Union[int, float]:
    '''Returns a configuration value_list.
    If the value_list is a list, returns the value_list at the specified index.
    If the index is not in range then it takes the last value in the value_list.
    '''

    value_list = config.get(key, default)
    if verbose:
        if key not in config:
            print(f'Key: {key} set to default={default}')

    if isinstance(value_list, list):
        try:
            return value_list[index]
        except IndexError:
            return value_list[-1]

    return value_list


def get_range_index(
    mini_cfg: dict, iteration: int, total_iterations: int
) -> int:
    '''Return the index of the switches corresponding to the current iteration
    number.

    Example:
    -------
    total_iteration = 10
    current_iteration = 7
    switches = [0, 1, 5]

    get_range_index(cfg, current_iteration, total_iteration) -> 2
    '''

    switches = mini_cfg.get(SWITCHES, [0])
    switches = switches + [total_iterations]

    for i in range(len(switches)-1):
        if switches[i] <= iteration < switches[i+1]:
            return i


def n_samples_factory(mini_cfg: dict) -> Callable[[int], int]:
    '''Creates a Callable[iterations] which returns the number of samples.'''

    def n_samples(iteration: int) -> int:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        return get_config_value(N_SAMPLES, mini_cfg[SAMPLES], range_index, default=None)

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        n = n_samples(ii)
        if n is None:
            raise ValueError(
                f"Number of samples at iteration {ii} should be set.")

    return n_samples


def sample_mode_factory(mini_cfg: dict) -> Callable[[int], str]:
    '''Creates a Callable[iterations] which returns the sample mode for
    nifty8.re.'''

    def sample_mode(iteration: int) -> str:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        return get_config_value(MODE, mini_cfg[SAMPLES], range_index, default='').lower()

    sample_keywords = [
        "linear_sample",
        "linear_resample",
        "nonlinear_sample",
        "nonlinear_resample",
        "nonlinear_update",
    ]
    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        t = sample_mode(ii)
        if t not in sample_keywords:
            raise ValueError(f"Unknown sample type: {t} at iteration {ii}, "
                             "known types: {sample_keywords}")

    return sample_mode


def _delta_logic(
        keyword: str,
        delta: dict,
        overwritten_value: Union[float, None],
        iteration: int,
        switches_index: int,
        ndof: Optional[int] = None
) -> float:
    """
    Calculates minimization config value if `delta` is in config.
    If the minimization value is already set at the given iteration,
    it will not be overwritten.

    Parameters
    ----------
    keyword : str
        Type of the delta logic ('kl', 'linear', 'nonlinear').
    delta : dict
        Configuration dictionary for delta values.
    overwritten_value : Union[float, None]
        Value to possibly be overwritten.
    iteration : int
        Current global iteration index.
    switches_index : int
        Index within the current `switches` range.
    ndof : Optional[int]
        Number of constrained degrees of freedom, required for
        minimization parameter recalculation with `delta`.

    Returns
    -------
    float
        Possibly recalculated minimization parameter value.

    Raises
    ------
    ValueError
        If required values are not set or if an unknown keyword is used.

    Examples
    --------
    >>> delta_config = {'delta': [0.1, 0.2, 0.3]}
    >>> _delta_logic('kl', delta_config, None, 0, 0, 10)
    1.0
    >>> _delta_logic('linear', delta_config, None, 0, 0, 10)
    0.1
    >>> _delta_logic('nonlinear', delta_config, None, 0, 0, None)
    0.1
    """

    if overwritten_value is not None:
        return overwritten_value

    params = {
        'kl': {'variable': 'absdelta', 'factor': ndof},
        'linear': {'variable': 'absdelta', 'factor': ndof / 10},
        'nonlinear': {'variable': 'xtol', 'factor': 1.0}
    }

    param = params.get(keyword)

    if param is None:
        raise ValueError(f"Unknown keyword: {keyword}.")

    if delta is None:
        raise ValueError(f'The {keyword} {param["variable"]} in iteration {iteration} '
                         f'is not set. A `delta` must be set in the config.')

    delta_value = get_config_value(DELTA_VALUE, delta, switches_index, default=None)

    if delta_value is None:
        raise ValueError(f'{keyword}: delta value must be set.')

    return_value = delta_value * param['factor']
    logging.info(f'it {iteration}: {keyword} {param["variable"]} set to {return_value}')
    return return_value


def linear_sample_kwargs_factory(
    mini_cfg: dict, delta: Optional[dict] = None, ndof: Optional[int] = None
) -> Callable[[int], dict]:
    """Creates a Callable[iterations] which returns linear sample kwargs for
    nifty8.re."""

    def lin_kwargs_getter(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])

        absdelta_name = f'{LINKEY}_{ABSDELTA}'
        miniter_name = f'{LINKEY}_{MINITER}'
        maxiter_name = f'{LINKEY}_{MAXITER}'

        minit = get_config_value(
            miniter_name, mini_cfg[SAMPLES], range_index, default=None)
        maxit = get_config_value(
            maxiter_name, mini_cfg[SAMPLES], range_index, default=None)
        absdelta = get_config_value(
            absdelta_name, mini_cfg[SAMPLES], range_index, default=None)

        absdelta = _delta_logic(
            'linear', delta, absdelta, iteration, range_index, ndof)

        return dict(
            cg_name=f'Lin: {absdelta}',
            cg_kwargs=dict(absdelta=absdelta, miniter=minit, maxiter=maxit))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        lin_config = lin_kwargs_getter(ii)
        absdelta = lin_config['cg_kwargs']['absdelta']
        if absdelta is None:
            raise ValueError(
                'Linear sample: Either delta or absdelta need to be set')

    return lin_kwargs_getter


def nonlinear_update_kwargs_factory(
    mini_cfg: dict, delta: Optional[dict] = None
) -> Callable[[int], dict]:
    '''Creates a Callable[iterations] which returns nonlinear sample kwargs for
    nifty8.re.'''

    def nonlinear_update_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])

        absdelta_name = f'{NONLIN}_{XTOL}'
        miniter_name = f'{NONLIN}_{MINITER}'
        maxiter_name = f'{NONLIN}_{MAXITER}'
        minit = get_config_value(
            miniter_name, mini_cfg[SAMPLES], range_index, default=None)
        maxit = get_config_value(
            maxiter_name, mini_cfg[SAMPLES], range_index, default=None)
        xtol = get_config_value(
            absdelta_name, mini_cfg[SAMPLES], range_index, default=None)

        xtol = _delta_logic('nonlinear', delta, xtol, iteration, range_index)

        cg_delta_name = f'{NONLIN_CG}_{ABSDELTA}'
        cg_atol_name = f'{NONLIN_CG}_{ATOL}'
        cg_miniter_name = f'{NONLIN_CG}_{MINITER}'
        cg_maxiter_name = f'{NONLIN_CG}_{MAXITER}'
        cg_delta = get_config_value(
            cg_delta_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_atol = get_config_value(
            cg_atol_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_minit = get_config_value(
            cg_miniter_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_maxit = get_config_value(
            cg_maxiter_name, mini_cfg[SAMPLES], range_index, default=None)

        nl_name = f'{NONLIN}'
        cg_name = f'{NONLIN_CG}'
        return dict(
            minimize_kwargs=dict(
                name=nl_name,
                xtol=xtol,
                miniter=minit,
                maxiter=maxit,
                cg_kwargs=dict(
                    name=cg_name,
                    absdelta=cg_delta,
                    atol=cg_atol,
                    miniter=cg_minit,
                    maxiter=cg_maxit
                )))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        non_linear_samples_dict = nonlinear_update_kwargs(ii)
        if non_linear_samples_dict['minimize_kwargs']['xtol'] is None:
            raise ValueError(
                f"nonlinear xtol at iteration {ii} should be set.")

    return nonlinear_update_kwargs


def kl_kwargs_factory(
    mini_cfg: dict, delta: Optional[dict] = None, ndof: Optional[int] = None
) -> Callable[[int], dict]:
    '''Creates a Callable[iterations] which returns kl minimization kwargs for
    nifyt8.re.'''

    def kl_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[KL_MINI], iteration, mini_cfg[N_TOTAL_ITERATIONS])

        absdelta_name = '_'.join((KL, ABSDELTA))
        miniter_name = '_'.join((KL, MINITER))
        maxiter_name = '_'.join((KL, MAXITER))
        minit = get_config_value(
            miniter_name, mini_cfg[KL_MINI], range_index, default=None)
        maxit = get_config_value(
            maxiter_name, mini_cfg[KL_MINI], range_index, default=None)
        absdelta = get_config_value(
            absdelta_name, mini_cfg[KL_MINI], range_index, default=None)

        absdelta = _delta_logic(
            'kl', delta, absdelta, iteration, range_index, ndof)

        return dict(
            minimize_kwargs=dict(
                name=f'{KL}',
                absdelta=absdelta,
                miniter=minit,
                maxiter=maxit,
                cg_kwargs=dict(name=f'{KL}CG')
            ))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        kl_kwargs_dict = kl_kwargs(ii)
        if kl_kwargs_dict['minimize_kwargs']['absdelta'] is None:
            raise ValueError(f"kl absdelta at iteration {ii} should be set.")

    return kl_kwargs


def calculate_constrained_ndof(likelihood: jft.Likelihood):
    n_dof_data = jft.size(likelihood.left_sqrt_metric_tangents_shape)
    n_dof_model = jft.size(likelihood.domain)
    return min(n_dof_model, n_dof_data)


class MinimizationParser():
    def __init__(self, config, ndof=None):
        delta = config.get(DELTA)

        if (delta is not None) and ndof is None:
            raise ValueError('Number of relevant degrees of freedom have to ',
                             'be set, to allow delta evalutation')

        self.n_samples = n_samples_factory(config)
        self.sample_mode = sample_mode_factory(config)
        self.draw_linear_kwargs = linear_sample_kwargs_factory(
            config, delta, ndof=ndof)
        self.nonlinear_update_kwargs = nonlinear_update_kwargs_factory(
            config, delta)
        self.kl_kwargs = kl_kwargs_factory(config, delta, ndof=ndof)
