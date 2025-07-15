# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

import os
from os.path import join
import pickle

import numpy as np
import nifty.re as jft


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
    # TODO replace with nifty.re.mean_and_std
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


def save_to_yaml(dict, filename, dir=None, verbose=True):
    """Convenience function to save dicts to yaml files.

    Parameters
    ----------
    dict: dictionary
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
        yaml.dump(dict, f)
    if verbose:
        print(f"Config file saved to: {join(dir, filename)}.")


def copy_config(filename, path_to_yaml_file=None,
                     output_dir=None, verbose=True):
    """Convenience function to copy yaml-config files.

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
