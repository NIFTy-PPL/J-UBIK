# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Any, Optional, Dict

import nifty.re as jft

# CONFIGURATION DEFAULTS
SWITCHES = 'switches'
SAMPLES = 'samples'
N_SAMPLES = 'n_samples'
N_TOTAL_ITERATIONS = 'n_total_iterations'
CONSTANTS = 'constants'
POINT_ESTIMATES = 'point_estimates'
CONST_KEYS_CONFIG_NAME = 'domain_keys'
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
CG_OPTIONS = (ABSDELTA, TOL, ATOL, MINITER, MAXITER, 'resnorm', 'norm_ord')
NEWTON_OPTIONS = (ABSDELTA, XTOL, MINITER, MAXITER, 'norm_ord',
                  'energy_reduction_factor')


def get_config_value(
    key: str,
    config: dict,
    index: int,
    default: Any,
    verbose: bool = False
) -> int | float:
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
    int | float
        The configuration value at the specified index or the default value.

    Raises
    ------
    ValueError
        If the value list is empty.

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

    if value_list is None and default is not None:
        return default

    if isinstance(value_list, list):
        if not value_list:
            raise ValueError(f'{key}: value schedule must not be empty.')
        try:
            value = value_list[index]
        except IndexError:
            value = value_list[-1]
        if value is None and default is not None:
            return default
        return value

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
    if (not isinstance(switches, list) or not switches
            or any(type(value) is not int for value in switches)
            or switches[0] != 0
            or any(a >= b for a, b in zip(switches, switches[1:]))
            or switches[-1] > total_iterations):
        raise ValueError('switches must be increasing integers starting at 0 '
                         'and within n_total_iterations.')
    switches = switches + [total_iterations]

    for i in range(len(switches)-1):
        if switches[i] <= iteration < switches[i+1]:
            return i

    if iteration >= total_iterations:
        return len(switches) - 1


def _delta_logic(
    keyword: str,
    delta: dict | None,
    config_value: float | None,
    iteration: int,
    delta_switches_index: int,
    ndof: Optional[int] = None,
    verbose: bool = True,
) -> float | None:
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
    config_value : float, None
        Config value, possibly to be overwritten.
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
        Possibly recalculated minimization parameter value, or None to inherit.

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

    if config_value is not None:
        return config_value

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
        return None

    delta_value = get_config_value(DELTA_VALUE, delta, delta_switches_index,
                                   default=None)

    if delta_value is None:
        raise ValueError(f'{keyword}: delta value must be set.')

    if param['factor'] is None:
        raise ValueError(f'{keyword}: n_dof must be set to derive a tolerance.')
    return_value = delta_value * param['factor']
    if verbose:
        jft.logger.info(f'it {iteration}: {keyword} {param["variable"]} '
                        f'set to {return_value}')
    return return_value


def _solver_kwargs(config, prefix, options, index):
    """Read scheduled options; omit None so NIFTy supplies its defaults."""
    kwargs = {}
    for option in options:
        value = get_config_value(f'{prefix}_{option}', config, index, None)
        if value is not None:
            kwargs[option] = value
    return kwargs


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

    # Checks whether `n_samples` is well-defined before inference and prints its
    # values at each iteration.
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

    # Checks whether the `sample_mode` is well-defined before inference and
    # prints its values at each iteration.
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
    Creates a callable that returns linear sample kwargs for `nifty.re` based
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
        If a delta-derived tolerance cannot be resolved.
    """

    def linear_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        delta_range_index = get_range_index(
            delta or {}, iteration, mini_cfg[N_TOTAL_ITERATIONS])
        cg_kwargs = _solver_kwargs(mini_cfg[SAMPLES], LIN, CG_OPTIONS,
                                   range_index)
        absdelta = _delta_logic(LIN, delta, cg_kwargs.get(ABSDELTA), iteration,
                                delta_range_index, ndof, verbose)
        if absdelta is not None:
            cg_kwargs[ABSDELTA] = absdelta
        return dict(
            # NIFTy currently takes the name from cg_kwargs.
            cg_name=None,
            cg_kwargs=dict(name=f'{LIN_NAME}_cg', **cg_kwargs))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        linear_kwargs(ii)
    return linear_kwargs


def nonlinearly_update_kwargs_factory(
    mini_cfg: dict,
    delta: Optional[dict] = None,
    verbose: bool = True
) -> Callable[[int], dict]:
    """
    Creates a callable that returns nonlinear sample kwargs for `nifty.re`
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
        If a delta-derived tolerance cannot be resolved.
    """

    def nonlinearly_update_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[SAMPLES], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        delta_range_index = get_range_index(
            delta or {}, iteration, mini_cfg[N_TOTAL_ITERATIONS])
        kwargs = _solver_kwargs(mini_cfg[SAMPLES], NONLIN, NEWTON_OPTIONS,
                                range_index)
        xtol = _delta_logic(NONLIN, delta, kwargs.get(XTOL), iteration,
                            delta_range_index, verbose=verbose)
        if xtol is not None:
            kwargs[XTOL] = xtol
        cg_kwargs = _solver_kwargs(mini_cfg[SAMPLES], NONLIN_CG, CG_OPTIONS,
                                   range_index)
        return dict(minimize_kwargs=dict(
            name=NONLIN_NAME, **kwargs,
            cg_kwargs=dict(name=NONLIN_CG, **cg_kwargs)))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        nonlinearly_update_kwargs(ii)
    return nonlinearly_update_kwargs


def kl_kwargs_factory(
    mini_cfg: dict,
    delta: Optional[Dict[str, float]] = None,
    ndof: Optional[int] = None,
    verbose: bool = True,
) -> Callable[[int], dict]:
    """
    Creates a callable that returns KL minimization kwargs for `nifty.re`
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
        If a delta-derived tolerance cannot be resolved.
    """

    def kl_kwargs(iteration: int) -> dict:
        range_index = get_range_index(
            mini_cfg[KL_MINI], iteration, mini_cfg[N_TOTAL_ITERATIONS])
        delta_range_index = get_range_index(
            delta or {}, iteration, mini_cfg[N_TOTAL_ITERATIONS])
        kwargs = _solver_kwargs(mini_cfg[KL_MINI], KL, NEWTON_OPTIONS,
                                range_index)
        absdelta = _delta_logic(KL, delta, kwargs.get(ABSDELTA), iteration,
                                delta_range_index, ndof, verbose)
        if absdelta is not None:
            kwargs[ABSDELTA] = absdelta
        cg_kwargs = _solver_kwargs(mini_cfg[KL_MINI], KL_CG, CG_OPTIONS,
                                   range_index)
        return dict(minimize_kwargs=dict(
            name=KL_NAME, **kwargs,
            cg_kwargs=dict(name=KL_CG, **cg_kwargs)))

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        kl_kwargs(ii)
    return kl_kwargs


def constants_factory(
    mini_cfg: dict,
    verbose: bool = True
) -> Callable[[int], tuple | None]:
    """
    Creates a Callable that returns a tuple of domain keys which should be kept
    constant during minimization at a given iteration.

    Parameters
    ----------
    mini_cfg : dict
        The configuration dictionary containing constant keys information.

    verbose : bool, optional
        If True, prints the constants for each iteration.

    Returns
    -------
    Callable[[int], tuple]
        A function that takes an iteration number and returns a tuple of constant
        domain keys (or None).

    Examples
    --------
    >>> mini_cfg = {'constants': {'switches': [0, 5],
    ...             'domain_keys': [['a', 'b'], None]},
    ...             'n_total_iterations': 7}
    >>> constants = constants_factory(mini_cfg)
    >>> constants(0)
    ('a', 'b')
    >>> constants(6)
    None
    """

    return _domain_keys_factory(mini_cfg, CONSTANTS, verbose)


def point_estimates_factory(
    mini_cfg: dict,
    verbose: bool = True
) -> Callable[[int], tuple | None]:
    """
    Creates a Callable that returns a tuple of domain keys which should be
    treated as point estimates during minimization at a given iteration.

    Parameters
    ----------
    mini_cfg : dict
        The configuration dictionary containing point estimate keys information.

    verbose : bool, optional
        If True, prints the point estimates for each iteration.

    Returns
    -------
    Callable[[int], tuple]
        A function that takes an iteration number and returns a tuple of point
        estimate domain keys (or None).
    """

    return _domain_keys_factory(mini_cfg, POINT_ESTIMATES, verbose)


def _domain_keys_factory(mini_cfg, section, verbose):
    section_cfg = mini_cfg.get(section)

    def domain_keys(iteration):
        if section_cfg is None:
            return None
        index = get_range_index(section_cfg, iteration,
                                mini_cfg[N_TOTAL_ITERATIONS])
        keys = get_config_value(CONST_KEYS_CONFIG_NAME, section_cfg, index, None)
        if keys is not None:
            if (not isinstance(keys, (list, tuple))
                    or any(not isinstance(key, str) for key in keys)):
                raise ValueError(f"{section}.domain_keys: expected a schedule "
                                 "of key lists, e.g. [['a', 'b'], None].")
            keys = tuple(keys)
        if verbose:
            jft.logger.info(f'it {iteration + 1}: {section} set to {keys}')
        return keys

    for ii in range(mini_cfg[N_TOTAL_ITERATIONS]):
        domain_keys(ii)
    return domain_keys


class MinimizationParser:
    """
    Parses a configuration to set up functions for generating minimization
    kwargs for the `nifty.re` different minimization modes.

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
    constants : Callable[[int], tuple | None]
        Function returning a tuple of constant domain keys for each iteration.
    point_estimates : Callable[[int], tuple | None]
        Function returning a tuple of point estimate domain keys for each
        iteration.
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
        self.constants = constants_factory(config, verbose=verbose)
        self.point_estimates = point_estimates_factory(config, verbose=verbose)
