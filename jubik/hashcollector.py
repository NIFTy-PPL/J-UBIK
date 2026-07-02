# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %

from os.path import join
import subprocess
from importlib import resources
import fnmatch
import os
import ast
import re
import json


def _get_git_hash_from_local_package(package_name, git_path=None):
    """
    Retrieve the latest Git commit hash for a local Python package.

    This function fetches the latest Git commit hash for a specified local
    Python package. It handles both editable and non-editable installations.

    Parameters:
    -----------
    package_name : str
        The name of the package for which to retrieve the Git commit hash.
    git_path : str, optional
        The path to the git repository. If not provided, it will try to find it.

    Returns:
    --------
    str
        The latest Git commit hash for the specified package.

    Raises:
    -------
    ValueError
        If the Git hash cannot be retrieved due to a command error or evaluation issue.
    FileNotFoundError
        If the .git directory or necessary files are not found.
    KeyError
        If the package mapping variable is not found in the editable path file.
    """
    def get_git_hash(path):
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=path).strip().decode('utf-8')
        except subprocess.CalledProcessError:
            raise ValueError('Failed to get the latest commit hash')

    # Get the distribution metadata for the package
    try:
        package_path = resources.files(package_name).__str__()
    except ModuleNotFoundError:
        raise FileNotFoundError(f"Package '{package_name}' not found")

    editable_path = None
    variable_value = None

    for file_name in os.listdir(package_path):
        if fnmatch.fnmatch(file_name, f"*{package_name}*.py"):
            editable_path = join(package_path, file_name)
            break

    if editable_path is not None:
        # Read the path to the source directory from the .egg-link file
        variable_name = 'MAPPING'
        pattern = re.compile(rf"^\s*{variable_name}\s*=\s*(.+)")

        try:
            with open(editable_path, 'r') as f:
                for line in f:
                    match = pattern.match(line)
                    if match:
                        value_str = match.group(1)
                        try:
                            variable_value = ast.literal_eval(value_str)
                        except (SyntaxError, ValueError):
                            raise ValueError(f"Could not evaluate the value of '{variable_name}'")
                        break
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{editable_path}' not found")

        if variable_value is not None and package_name in variable_value:
            source_path = variable_value[package_name]
        else:
            raise KeyError(f"Variable '{variable_name}' not found or '{package_name}' "
                           f"not in mapping.")
    else:
        # Use the distribution location for non-editable installations
        source_path = package_path

    # Get the parent directory
    source_path = os.path.dirname(source_path)

    # Attempt to fix path
    if source_path.endswith("site-packages"):
        source_path = join(source_path, package_name)

    # Construct the path to the .git directory
    git_dir = join(source_path, '.git')

    if os.path.isdir(git_dir):
        return get_git_hash(source_path)
    elif git_path is not None:
        return get_git_hash(git_path)
    else:
        raise FileNotFoundError('No .git directory found. \n'
                                'Please provide the path to the git '
                                'repository for the given package.')


def save_local_packages_hashes_to_txt(packages_names,
                                      filename,
                                      paths_to_git=None,
                                      verbose=True):
    """
    Save the latest Git hashes of local packages to a text file.

    This function retrieves the latest Git commit hashes for a list of
    specified local Python packages and saves these hashes to a text file
    in JSON format. The output file will contain a dictionary where the
    keys are package names and the values are the corresponding Git hashes.

    Parameters:
    -----------
    packages_names : list of str
        A list of package names for which the Git hashes are to be retrieved.
    filename : str
        The name of the file where the Git hashes will be saved.
        The output file will be in JSON format, making it easy to
        read and parse.
    paths_to_git : list of str, optional
        A list of paths to the git repositories for the specified packages.
        If not provided or None, the function will attempt to retrieve
        the paths from the editable path file.
    verbose : bool, optional
        If True, print out the progress of the function.

    Returns:
    --------
    None
        This function does not return a value.
        It writes the output to the specified file.

    Example:
    --------
    >>> save_local_packages_hashes_to_txt(['example_package1',
                                         'example_package2'],
                                         'hashes.txt')
    Processing package: example_package1
    Successfully retrieved hash for example_package1: a1b2c3d4e5
    Processing package: example_package2
    Error processing package example_package2: Package 'example_package2' not found
    Hashes have been saved to package_hashes.txt

    The above example retrieves the Git hashes for 'example_package1'
    and 'example_package2', handles any errors encountered, and saves
    the results to 'package_hashes.txt'.
    """
    hashes = {}
    for it, package_name in enumerate(packages_names):
        try:
            if verbose:
                print(f"Processing package: {package_name}")
            if paths_to_git is not None and paths_to_git[it] is not None:
                git_hash = _get_git_hash_from_local_package(package_name,
                                                            git_path=paths_to_git[it])
            else:
                git_hash = _get_git_hash_from_local_package(package_name)
            hashes[package_name] = git_hash
            if verbose:
                print(f"Successfully retrieved hash for {package_name}: {git_hash}")
        except (ValueError, FileNotFoundError, KeyError) as e:
            if verbose:
                print(f"Error processing package {package_name}:\n")
            raise e

    with open(filename, 'w') as f:
        json.dump(hashes, f, indent=4)

    print(f"Hashes have been saved to {filename}.")
