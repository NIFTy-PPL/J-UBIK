# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Union, Any, Optional, Dict

import nifty8.re as jft

# CONFIGURATION CONSTANTS
SWITCHES = 'switches'
SAMPLES = 'samples'
N_SAMPLES = 'n_samples'
N_TOTAL_ITERATIONS = 'n_total_iterations'
MODE = 'mode'
DELTA = 'delta'
LIN = 'lin'
LIN_NAME = 'lin sampling'
ABSDELTA = 'absdelta'
MINITER = 'miniter'
MAXITER = 'maxiter'
DELTA_VALUE = 'values'
TOL = 'tol'
ATOL = 'atol'
XTOL = 'xtol'
NONLIN = 'nonlin'
NONLIN_NAME = 'nonlin sampling'
NONLIN_CG = 'nonlin_cg'
KL_MINI = 'kl_minimization'
KL = 'kl'
KL_NAME = 'kl'
KL_CG = 'kl_cg'


def get_config_value(
    key: str,
    config: dict,
    index: int, default: Any,
    verbose: bool = False
) -> Union[int, float]:
    """
    Returns a configuration value from a list or a default value.

    If the value is a list, returns the value at the specified index.
    If the index is out of range, returns the last value in the list.

    Parameters
    ----------
    key : str
        The configuration key to look up.
    config : dict
        The configuration dictionary.
    index : int
        The index to access within the list of values.
    default : Any
        The default value to return if the key is not in the config.
    verbose : bool, optional
        If True, prints a message when the key is not found (default is False).

    Returns
    -------
    Union[int, float]
        The configuration value at the specified index or the default value.

    Raises
    ------
    IndexError
        If the index is out of range for the value list.

    Examples
    --------
    >>> config = {'key1': [1, 2, 3]}
    >>> get_config_value('key1', config, 0, 0)
    1
    >>> get_config_value('key1', config, 3, 0)
    3
    >>> get_config_value('key2', config, 0, 0)
    0
    """

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
    mini_cfg: dict,
    iteration: int,
    total_iterations: int
) -> int:
    """
    Returns the index of the switches corresponding to the current iteration
    number.

    Parameters
    ----------
    mini_cfg : dict
        The configuration dictionary containing switches.
    iteration : int
        The current iteration number.
    total_iterations : int
        The total number of iterations.

    Returns
    -------
    int
        The index within the switches corresponding to the current iteration
        number.

    Examples
    --------
    >>> mini_cfg = {'switches': [0, 1, 5]}
    >>> get_range_index(mini_cfg, 7, 10)
    2
    >>> get_range_index(mini_cfg, 0, 10)
    0
    >>> get_range_index(mini_cfg, 5, 10)
    2
    """

    iteration += 1  # iteration index changes during OptVI update

    switches = mini_cfg.get(SWITCHES, [0])
    if switches is None:
        switches = [0]
    switches = switches + [total_iterations]

    for i in range(len(switches)-1):
        if switches[i] <= iteration < switches[i+1]:
            return i

    if iteration == total_iterations:
        return len(switches) - 1
    else:
        raise ValueError(f'Iteration {iteration} is out of range.')


def _delta_logic(
    keyword: str,
    delta: dict,
    overwritten_value: Union[float, None],
    iteration: int,
    delta_switches_index: int,
    ndof: Optional[int] = None,
    verbose: bool = True
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
    delta_switches_index : int
        Index within the current `switches` range for the `delta` parameter.
    ndof : Optional[int]
        Number of constrained degrees of freedom, required for
        minimization parameter recalculation with `delta`.
    verbose : Optional[bool]
        If True, prints the newly overwritten value.

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

    iteration += 1  # iteration index changes during OptVI update

    params = {
        KL: {'variable': ABSDELTA, 'factor': ndof},
        LIN: {'variable': ABSDELTA, 'factor': ndof / 10
        if ndof is not None else ndof},
        NONLIN: {'variable': XTOL, 'factor': 1.0}
    }

    param = params.get(keyword)

    if param is None:
        raise ValueError(f"Unknown keyword: {keyword}.")

    if delta is None:
        raise ValueError(f'The {keyword} {param["variable"]} in '
                         f'iteration {iteration} is not set. '
                         f'A `delta` must be set in the config.')

    delta_value = get_config_value(DELTA_VALUE, delta, delta_switches_index,
                                   default=None)

    if delta_value is None:
        raise ValueError(f'{keyword}: delta value must be set.')

    return_value = delta_value * param['factor']
    if verbose:
        jft.logger.info(f'it {iteration}: {keyword} {param["variable"]} '
                        f'set to {return_value}')
    return return_value


def n_samples_factory(
    mini_cfg: dict,
) -> Callable[[int], int]:
    """
    Creates a Callable that returns the number of samples for a given iteration.

    Parameters
    ----------
    mini_cfg : dict
        The configuration dictionary containing n_samples information.

    Returns
    -------
    Callable[[int], int]
        A function that takes an iteration number and returns the number
        of samples.

    Raises
    ------
    ValueError
        If the number of samples at any iteration is not set.

    Examples
    --------
    >>> mini_cfg = {'samples': {'switches': [0, 5], 'n_samples': [10, 20]},
    ...             'n_total_iterations': 7}
    >>> n_samples = n_samples_factory(mini_cfg)
    >>> n_samples(0)
    10
    >>> n_samples(6)
    20
    """

    def n_samples(iteration: int) -> int:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        return get_config_value(N_SAMPLES, mini_cfg[SAMPLES], range_index,
                                default=None)

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        n = n_samples(ii)
        if n is None:
            raise ValueError(
                f"Number of samples at iteration {ii+1} needs to be set.")

    return n_samples


def sample_mode_factory(
    mini_cfg: dict
) -> Callable[[int], str]:
    """
    Creates a Callable that returns the sample mode for a given iteration.

    Parameters
    ----------
    mini_cfg : dict
        The configuration dictionary containing sample information.

    Returns
    -------
    Callable[[int], str]
        A function that takes an iteration number and returns the sample mode.

    Raises
    ------
    ValueError
        If an unknown sample type is encountered.
    """

    def sample_mode(iteration: int) -> str:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        return get_config_value(MODE, mini_cfg[SAMPLES], range_index,
                                default='').lower()

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
            raise ValueError(f"Unknown sample type: {t} at iteration {ii+1}, "
                             "known types: {sample_keywords}")

    return sample_mode


def linear_sample_kwargs_factory(
    mini_cfg: dict,
    delta: Optional[dict] = None,
    ndof: Optional[int] = None,
    verbose: bool = True
) -> Callable[[int], dict]:
    """
    Creates a callable that returns linear sample kwargs for `nifty8.re` based
    on the current iteration.

    Parameters
    ----------
    mini_cfg : dict
        Configuration dictionary containing settings for the samples and the
        total number of iterations.
    delta : dict, optional
        Dictionary containing delta values for adjustment.
    ndof : int, optional
        Number of degrees of freedom used in the delta logic.
    verbose : bool, optional
        If True, the delta logic will be verbose.

    Returns
    -------
    Callable[[int], dict]
        A function that takes the current iteration as input and returns a
        dictionary of linear sample kwargs.

    Raises
    ------
    ValueError
        If the `absdelta` for any iteration is not set.
    """

    def linear_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        delta_range_index = get_range_index(
            mini_cfg[DELTA], iteration, mini_cfg[N_TOTAL_ITERATIONS]
        )

        absdelta_name = f'{LIN}_{ABSDELTA}'
        miniter_name = f'{LIN}_{MINITER}'
        maxiter_name = f'{LIN}_{MAXITER}'
        tol_name = f'{LIN}_{TOL}'
        atol_name = f'{LIN}_{ATOL}'

        minit = get_config_value(
            miniter_name, mini_cfg[SAMPLES], range_index, default=None)
        maxit = get_config_value(
            maxiter_name, mini_cfg[SAMPLES], range_index, default=None)
        absdelta = get_config_value(
            absdelta_name, mini_cfg[SAMPLES], range_index, default=None)
        tol = get_config_value(
            tol_name, mini_cfg[SAMPLES], range_index, default=None)
        atol = get_config_value(
            atol_name, mini_cfg[SAMPLES], range_index, default=None)

        absdelta = _delta_logic(LIN, delta, absdelta, iteration,
                                delta_range_index, ndof, verbose)

        return dict(
            cg_name=LIN_NAME,
            cg_kwargs=dict(
                name=None,
                absdelta=absdelta,
                tol=tol,
                atol=atol,
                miniter=minit,
                maxiter=maxit)
            )

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        lin_config = linear_kwargs(ii)
        absdelta = lin_config['cg_kwargs']['absdelta']
        if absdelta is None:
            raise ValueError(f'Linear `absdelta` at iteration {ii+1} '
                             f'needs to be set.')

    return linear_kwargs


def nonlinearly_update_kwargs_factory(
    mini_cfg: dict,
    delta: Optional[dict] = None,
    verbose: bool = True
) -> Callable[[int], dict]:
    """
    Creates a callable that returns nonlinear sample kwargs for `nifty8.re`
    based on the current iteration.

    Parameters
    ----------
    mini_cfg : dict
        Configuration dictionary containing settings for the samples and the
        total number of iterations.
    delta : dict, optional
        Dictionary containing delta values for adjustment.
    verbose : bool, optional
        If True, the delta logic will be verbose.

    Returns
    -------
    Callable[[int], dict]
        A function that takes the current iteration as input and returns a
        dictionary of nonlinear sample kwargs.

    Raises
    ------
    ValueError
        If the `xtol` for any iteration is not set.
    """

    def nonlinearly_update_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        delta_range_index = get_range_index(
            mini_cfg[DELTA], iteration, mini_cfg[N_TOTAL_ITERATIONS]
        )

        absdelta_name = f'{NONLIN}_{XTOL}'
        miniter_name = f'{NONLIN}_{MINITER}'
        maxiter_name = f'{NONLIN}_{MAXITER}'
        minit = get_config_value(
            miniter_name, mini_cfg[SAMPLES], range_index, default=None)
        maxit = get_config_value(
            maxiter_name, mini_cfg[SAMPLES], range_index, default=None)
        xtol = get_config_value(
            absdelta_name, mini_cfg[SAMPLES], range_index, default=None)

        xtol = _delta_logic(NONLIN, delta, xtol, iteration,
                            delta_range_index, verbose)

        cg_delta_name = f'{NONLIN_CG}_{ABSDELTA}'
        cg_tol_name = f'{NONLIN_CG}_{TOL}'
        cg_atol_name = f'{NONLIN_CG}_{ATOL}'
        cg_miniter_name = f'{NONLIN_CG}_{MINITER}'
        cg_maxiter_name = f'{NONLIN_CG}_{MAXITER}'

        cg_delta = get_config_value(
            cg_delta_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_tol = get_config_value(
            cg_tol_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_atol = get_config_value(
            cg_atol_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_minit = get_config_value(
            cg_miniter_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_maxit = get_config_value(
            cg_maxiter_name, mini_cfg[SAMPLES], range_index, default=None)

        return dict(
            minimize_kwargs=dict(
                name=NONLIN_NAME,
                xtol=xtol,
                miniter=minit,
                maxiter=maxit,
                cg_kwargs=dict(
                    name=NONLIN_CG,
                    absdelta=cg_delta,
                    tol=cg_tol,
                    atol=cg_atol,
                    miniter=cg_minit,
                    maxiter=cg_maxit
                )))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        non_linear_samples_dict = nonlinearly_update_kwargs(ii)
        if non_linear_samples_dict['minimize_kwargs']['xtol'] is None:
            raise ValueError(
                f"nonlinear `xtol` at iteration {ii+1} "
                f"needs to be set.")

    return nonlinearly_update_kwargs


def kl_kwargs_factory(
    mini_cfg: dict,
    delta: Optional[Dict[str, float]] = None,
    ndof: Optional[int] = None,
    verbose: bool = True,
) -> Callable[[int], dict]:
    """
    Creates a callable that returns KL minimization kwargs for `nifty8.re`
    based on the current iteration.

    Parameters
    ----------
    mini_cfg : dict
        Configuration dictionary containing settings for the KL minimization.
    delta : dict, optional
        Dictionary containing delta values for adjustment.
    ndof : int, optional
        Degrees of freedom parameter.
    verbose : bool, optional
        If True, the delta logic will be verbose.

    Returns
    -------
    Callable[[int], dict]
        A function that takes the current iteration as input and returns
        a dictionary of KL minimization kwargs.

    Raises
    ------
    ValueError
        If the `absdelta` for any iteration is not set.
    """

    def kl_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[KL_MINI], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        delta_range_index = get_range_index(
            mini_cfg[DELTA], iteration, mini_cfg[N_TOTAL_ITERATIONS]
        )

        absdelta_name = f'{KL}_{ABSDELTA}'
        miniter_name = f'{KL}_{MINITER}'
        maxiter_name = f'{KL}_{MAXITER}'
        xtol_name = f'{KL}_{XTOL}'

        minit = get_config_value(
            miniter_name, mini_cfg[KL_MINI], range_index, default=None)
        maxit = get_config_value(
            maxiter_name, mini_cfg[KL_MINI], range_index, default=None)
        absdelta = get_config_value(
            absdelta_name, mini_cfg[KL_MINI], range_index, default=None)
        xtol = get_config_value(
            xtol_name, mini_cfg[KL_MINI], range_index, default=None)

        absdelta = _delta_logic(KL, delta, absdelta, iteration,
                                delta_range_index, ndof, verbose)

        cg_absdelta_name = f'{KL_CG}_{ABSDELTA}'
        cg_atol_name = f'{KL_CG}_{ATOL}'
        cg_miniter_name = f'{KL_CG}_{MINITER}'
        cg_maxiter_name = f'{KL_CG}_{MAXITER}'
        cg_atol = get_config_value(
            cg_atol_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_absdelta = get_config_value(
            cg_absdelta_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_minit = get_config_value(
            cg_miniter_name, mini_cfg[SAMPLES], range_index, default=None)
        cg_maxit = get_config_value(
            cg_maxiter_name, mini_cfg[SAMPLES], range_index, default=None)

        return dict(
            minimize_kwargs=dict(
                name=KL_NAME,
                absdelta=absdelta,
                miniter=minit,
                maxiter=maxit,
                xtol=xtol,
                cg_kwargs=dict(
                    name=KL_CG,
                    absdelta=cg_absdelta,
                    atol=cg_atol,
                    miniter=cg_minit,
                    maxiter=cg_maxit
                )
            ))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        kl_kwargs_dict = kl_kwargs(ii)
        if kl_kwargs_dict['minimize_kwargs']['absdelta'] is None:
            raise ValueError(f"kl `absdelta` at iteration {ii+1} "
                             f"needs to be set.")

    return kl_kwargs


class MinimizationParser:
    """
    Parses a configuration to set up functions for generating minimization
    kwargs for the `nifty8.re` different minimization modes.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing settings for minimization.
    n_dof : int, optional
        Number of degrees of freedom relevant for the calculations.
    verbose : bool, optional
        Whether to print verbose information.

    Raises
    ------
    ValueError
        If `delta` is provided in the configuration but `n_dof` is not set.

    Attributes
    ----------
    n_samples : Callable[[int], int]
        Function returning the number of samples for each iteration.
    sample_mode : Callable[[int], str]
        Function returning the sample mode for each iteration.
    draw_linear_kwargs : Callable[[int], dict]
        Function returning linear sample kwargs for each iteration.
    nonlinearly_update_kwargs : Callable[[int], dict]
        Function returning nonlinear update kwargs for each iteration.
    kl_kwargs : Callable[[int], dict]
        Function returning KL minimization kwargs for each iteration.
    """

    def __init__(self, config, n_dof=None, verbose=True):
        delta = config.get(DELTA)

        if (delta is not None) and n_dof is None:
            raise ValueError('Number of relevant degrees of freedom must '
                             'be set to allow recalculating minimization '
                             'values with `delta`.')

        self.n_samples = n_samples_factory(config)
        self.sample_mode = sample_mode_factory(config)
        self.draw_linear_kwargs = linear_sample_kwargs_factory(
            config, delta, ndof=n_dof, verbose=verbose)
        self.nonlinearly_update_kwargs = nonlinearly_update_kwargs_factory(
            config, delta, verbose=verbose)
        self.kl_kwargs = kl_kwargs_factory(config, delta,
                                           ndof=n_dof, verbose=verbose)
