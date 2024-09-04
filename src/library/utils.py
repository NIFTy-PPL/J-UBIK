import ast
import fnmatch
import json
import os
import re
import subprocess
import pickle
from importlib import resources
from os.path import join
from warnings import warn

import nifty8 as ift
import nifty8.re as jft
import numpy as np
import scipy


def load_from_pickle(file_path):
    """Load an object from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the pickle file (.pkl) from which the object will be loaded.

    Returns
    -------
    obj : object
        The object loaded from the pickle file. The type of this object can vary
        depending on what was originally serialized into the pickle file.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_to_pickle(obj, file_path):
    """Save an object to a pickle file.

    Parameters
    ----------
    obj : object
        The object saved to the pickle file. The type of this object can vary
        depending on what shall be saved to the pickle file.
    file_path : string
        Path to data file (.pkl)
    """
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def add_models(m1, m2):
    """Summation of two models.

    Builds a model that takes two models m1 and m2 and adds their results.

    Parameters
    ----------
    m1: jft.Model
    m2: jft.Model

    Returns
    -------
    sum: jft.Model
    """
    domain = m1.domain
    domain.update(m2.domain)
    return jft.Model(lambda x: m1(x) + m2(x), domain=domain)


def add_functions(f1, f2):
    """Summation of two functions.

    Builds a function that takes two functions f1 and f2
    and adds their results.

    Parameters
    ----------
    f1: callable
    f2: callable

    Returns
    -------
    sum: callable
    """
    def function(x):
        return f1(x) + f2(x)
    return function


def get_stats(sample_list, func):
    # TODO replace with jft.mean_and_std
    """Return stats(mean and std) for sample_list.

    Parameters:
    ----------
    sample_list: list of posterior samples
    func: callable
    """
    f_s = np.array([func(s) for s in sample_list])
    return f_s.mean(axis=0), f_s.std(axis=0, ddof=1)


def get_config(path_to_yaml_file):
    """Convenience function for loading yaml-config files.

    Parameters
    ----------
    path_to_yaml_file: str,
        The location of the config file

    Returns
    -------
    dictionary
        a dictionary containing all the information stored in the config.yaml

    """
    import yaml
    with open(path_to_yaml_file, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg


def save_config(config, filename, dir=None, verbose=True):
    """Convenience function to save yaml-config files.

    Parameters
    ----------
    config: dictionary
        dictionary containing the config information
    filename: str
        Name of the config file to be saved.
    dir: str
        Location where the filename.yaml should be saved.
    verbose: bool
        If true, print a message when the config file is saved.
    """
    import yaml
    if dir is not None:
        create_output_directory(dir)
    with open(join(dir, filename), "w") as f:
        yaml.dump(config, f)
    if verbose:
        print(f"Config file saved to: {join(dir, filename)}.")


def save_config_copy(filename, path_to_yaml_file=None,
                     output_dir=None, verbose=True):
    # TODO can this be replaced by save config?
    """Convenience function to save yaml-config files.

    Parameters
    ----------
    filename: str
        Name of the config file to be copied.
    path_to_yaml_file: str
        The location of the config file.
    output_dir: str
        Location to which the filename.yaml should be copied.
    verbose: bool
        If true, print a message when the config file is saved.
    """
    if output_dir is not None:
        create_output_directory(output_dir)
    current_filename = filename
    if path_to_yaml_file is not None:
        current_filename = join(path_to_yaml_file, current_filename)
    os.popen(f'cp {current_filename} {join(output_dir, filename)}')
    if verbose:
        print(f"Config file saved to: {join(output_dir, filename)}.")


def create_output_directory(directory_name):
    """Convenience function to create directories.
    # TODO: is this needed?
    Parameters
    ----------
    directory_name: str
        path of the directory which will be created

    Returns:
    --------
    directory_name: str
    """
    os.makedirs(directory_name, exist_ok=True)
    return directory_name


def get_gaussian_psf(op, var):
    """
    # TODO ask matteani if he uses that one
    Builds a convolution operator which can be applied to an nifty8.Operator.
    It convolves the result of the operator with a Gaussian Kernel.

    Parameters
    ---------
    op: nifty8.Operator
        The Operator to which we'll apply the convolution
    var: float
        The variance of the Gaussian Kernel

    Returns
    -------
    nifty8.Operator
    """
    # FIXME: cleanup -> refactor into get_gaussian_kernel
    dist_x = op.target[0].distances[0]
    dist_y = op.target[0].distances[1]

    # Periodic Boundary conditions
    x_ax = np.arange(op.target[0].shape[0])
    x_ax = np.minimum(x_ax, op.target[0].shape[0] - x_ax) * dist_x
    y_ax = np.arange(op.target[0].shape[1])
    y_ax = np.minimum(y_ax, op.target[0].shape[1] - y_ax) * dist_y

    center = (0, 0)
    x_ax -= center[0]
    y_ax -= center[1]
    X, Y = np.meshgrid(x_ax, y_ax, indexing='ij')

    var *= op.target[0].scalar_dvol  # ensures that the variance parameter is specified with respect to the

    # normalized psf
    log_psf = - (0.5 / var) * (X ** 2 + Y ** 2)
    log_kernel = ift.makeField(op.target[0], log_psf)
    log_kernel = log_kernel - np.log(log_kernel.exp().integrate().val)

    # p = ift.Plot()
    # import matplotlib.colors as colors
    # p.add(log_kernel.exp(), norm=colors.SymLogNorm(linthresh=10e-8))
    # p.output(nx=1)

    conv_op = get_fft_psf_op(log_kernel.exp(), op.target)
    return conv_op


def coord_center(side_length, side_n):
    """
    Calculates the indices of the centers of the n**2 patches
    for a quadratical domain with a certain side length
    # TODO hide from scope

    Parameters
    ----------
    side_length: int
        length of one side
    side_n: int
        number of patches along one side

    Returns
    -------
    Array
    """
    tdx = tdy = side_length // side_n
    xc = np.arange(tdx // 2, tdx * side_n, tdx)
    yc = np.arange(tdy // 2, tdy * side_n, tdy)
    co = np.array(np.meshgrid(xc, yc)).reshape(2, -1)
    res = np.ravel_multi_index(co, [side_length, side_length])
    return res


def _check_type(arg, type, name=''):
    if arg is None:
        pass
    elif isinstance(arg, list):
        if not isinstance(arg[0], type):
            return TypeError(
                "The arguments of the \"{}\" list must be of type {}.".format(name, str(type)))
        else:
            pass
    elif not isinstance(arg, type):
        print("arg:", arg)
        raise TypeError("The \"{}\" argument must be of type {}.".format(name, str(type)))
