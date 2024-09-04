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





def save_to_fits(sample_list,
                 file_name_base,
                 op=None,
                 samples=False,
                 mean=False,
                 std=False,
                 overwrite=False,
                 obs_type="SF"):
    """Write sample list to FITS file.

    This function writes properties of a sample list to a FITS file according
    to the obs_type and based on the nifty8 function save_to_fits by P.Arras

    Parameters
    ----------
    file_name_base : str
        File name base of output FITS file, i.e. without `.fits` extension.
    op : callable or None
        Callable that is applied to each item in the :class:`SampleListBase`
        before it is returned. Can be an
        :class:`~nifty8.operators.operator.Operator` or any other callable
        that takes a :class:`~nifty8.field.Field` as an input. Default:
        None.
    samples : bool
        If True, samples are written into fits file.
    mean : bool
        If True, mean of samples is written into fits file.
    std : bool
        If True, standard deviation of samples is written into fits file.
    overwrite : bool
        If True, a potentially existing file with the same file name as
        `file_name`, is overwritten.
    obs_type : string or None
        Describes the observation type. currently possible obs_types are
        [CMF (Chandra Multifrequency), EMF (Erosita Multifrequency),
        RGB and SF (Single Frequency]. The default observation is of type SF.
        In the case of the type "RGB", the binning is automatically done
        by jubik into equally sized bins.
    """
    if not (samples or mean or std):
        raise ValueError("Neither samples nor mean nor standard deviation shall be written.")

    if mean or std:
        m, s = sample_list.sample_stat(op)

    if obs_type == "RGB":
        m = energy_binning(m, energy_bins=3)
        s = energy_binning(s, energy_bins=3)
    if mean:
        save_rgb_image_to_fits(m, file_name_base + "_mean", overwrite,
                               sample_list.MPI_master)
    if std:
        save_rgb_image_to_fits(s, file_name_base + "_std", overwrite,
                               sample_list.MPI_master)
    if samples:
        for ii, ss in enumerate(sample_list.iterator(op)):
            if obs_type == "RGB":
                ss = energy_binning(ss, energy_bins=3)
            save_rgb_image_to_fits(ss, file_name_base + f"_sample_{ii}",
                                   overwrite, sample_list.MPI_master)


def save_rgb_image_to_fits(fld, file_name, overwrite, MPI_master):
    """
    Takes a field with three energy bins and writes three according fits-files
    Parameters
    ----------
    fld: ift.Field
        Field with three energy bins
    file_name: str
        Base of name of output file
    overwrite: Bool
        If True overwrite existing files, if False do not overwrite existing files
    MPI_master: MPI comm

    Returns
    ----------
    None
    """
    import astropy.io.fits as pyfits
    from astropy.time import Time
    import time
    domain = fld.domain
    if not isinstance(domain, ift.DomainTuple) or len(domain[0].shape) !=2:
        raise ValueError(f"Expected DomainTuple with the first space being a 2-dim RGSpace, but got {domain}")
    if len(domain) == 2:
        if fld.shape[2] != 3:
            raise NotImplementedError("Energy direction has to be binned to 3 to create an RGB image. "
                                        f"Current number of energy bins:\n{fld.shape[2]}")
        npix_e = fld.shape[2]
        color_dict = {0: "red", 1: "green", 2: "blue"}
    elif len(domain) == 1:
        npix_e = 1
        color_dict = {0: "uni"}
    else:
        raise NotImplementedError
    # FIXME: Header improvement
    h = pyfits.Header()
    h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]
    h["CRVAL1"] = h["CRVAL2"] = 0  # coordinate value at reference point
    h["CRPIX1"] = h["CRPIX2"] = 0  # pixel coordinate of the reference point
    h["CUNIT1"] = h["CUNIT2"] = "arcsec"
    h["CDELT1"], h["CDELT2"] = -domain[0].distances[0], domain[0].distances[1] # coordinate increment
    h["CTYPE1"] = "RA" # axis type
    h["CTYPE2"] = "DEC"
    h["EQUINOX"] = 2000
    if MPI_master:
        for i in range(npix_e):
            if npix_e > 1:
                hdu = pyfits.PrimaryHDU(fld.val[:, :, i], header=h)
            else:
                hdu = pyfits.PrimaryHDU(fld.val, header=h)
            hdulist = pyfits.HDUList([hdu])
            file_name_colour = f"{file_name}_{color_dict[i]}.fits"
            hdulist.writeto(file_name_colour, overwrite=overwrite)
            print(f"RGB image saved as {file_name_colour}.")


def energy_binning(fld, energy_bins):
    """
    Takes a field with an arbitrary number of energy bins and reshapes it into
    a field with three energy-bins. Parameters. If the field has less than
    3 energy-bins the field is padded with a constant value. If the field
    has 3 energy bins, nothing happens and if the field has more than 3 energy
    bins the array is rebinned to 3 equally sized energy bins.

    Parameters
    ----------
    fld: ift.Field
        Field with energy direction
    energy_bins: int
        Number of energy bins the field should be reshaped to

    Return
    ----------
    fld: ift.Field
        Field with changed number of energy bins

    Note
    ----
    If the number of energy-bins divided by 3 is not an integer,
    the last bin will be bigger.
    """
    domain = fld.domain
    arr = fld.val
    shape = [i for i in arr.shape]
    new_shape = shape[:2]
    new_shape.append(energy_bins)
    new_domain = ift.DomainTuple.make((domain[0], ift.RGSpace(energy_bins)))
    aux_arrs = []
    binned_array = arr
    if shape[2] < energy_bins:
        binned_array = np.pad(arr, [(0, 0), (0, 0), (0, (energy_bins - shape[2]))], mode='constant')
    if shape[2] > energy_bins:
        bins = np.arange(0, shape[2] + 1, shape[2] / energy_bins)
        for i in range(len(bins) - 1):
            bin1 = int(bins[i])
            bin2 = int(bins[i + 1])
            aux_arrs.append(np.sum(arr[:, :, bin1:bin2], axis=2))
        binned_array = np.stack(aux_arrs, axis=2)
    binned_field = ift.Field.from_raw(new_domain, binned_array)
    return binned_field

# TODO do we need this?
def transform_loglog_slope_pars(slope_pars):
    """Transform slope parameters from log10/log10 to ln/log10 space

    Parameters
    -----------
    slope_pars: numpy.array

    Returns
    -------
    numpy.array
    """
    res = slope_pars.copy()
    res['mean'] = (res['mean'] + 1) * np.log(10)
    res['sigma'] *= np.log(10)
    return res


# TODO transfer this knowledge to jubix / likelihood
class _IGLikelihood(ift.EnergyOperator):
    """
    Functional form of the Inverse-Gamma distribution. Can be used for
    Equal-likelihood-optimization.

    Notes:
    ------
        This implementation is only designed for a point source component over
        a single-frequency sky.
    """
    # TODO: Build MF-version

    def __init__(self, data, alpha, q):
        """
        Constructs an EnergyOperator specifially for InverseGamma Likelihoods.

        Parameters
        ----------
        data: nifty8.Field
        alpha: float
        q: float
        """
        self._domain = ift.makeDomain(data.domain)
        shift = ift.Adder(data) @ ift.ScalingOperator(self._domain, -1)
        dummy = ift.ScalingOperator(self._domain, 1)
        self._q = ift.ScalingOperator(self._domain, float(q))
        self._apw = ift.ScalingOperator(self._domain, float(alpha + 1.))
        op = self._q @ dummy.ptw('reciprocal') + self._apw @ dummy.ptw('log')
        self._op = (op @ shift.ptw('abs')).sum()

    def apply(self, x):
        self._check_input(x)
        res = self._op(x)
        if not x.want_metric:
            return res
        raise NotImplementedError

# TODO to likelihood
def get_equal_lh_transition(sky, diffuse_sky, point_dict, transition_dict,
                            point_key = 'point_sources', stiffness = 1E6,
                            red_factor = 1E-3):
    """
    Performs a likelihood (i.E. input sky) invariant transition between the
    dofs of a diffuse component and point sources. Assumes `sky`to be composed
    as
        sky = diffuse_sky(xi_diffuse) + point_sky(xi_point)

    where `(..._)sky` are `nifty` Operators and `xi` are the standard dofs of
    the components. The operator `point_sky` is assumed to be a generative
    process for an Inverse-Gamma distribution matching the convention of
    `_IGLikelihood`.

    Parameters:
    -----------
        sky: nifty8.Operator
            Generative model for the sky consisting of a point source component
            and another additive component `diffuse_sky`.
        diffuse_sky: nifty8.Operator
            Generative model describing only the diffuse component.
        point_dict: dict of float
            Dictionary containing the Inverse-Gamma parameters `alpha` and `q`.
        transition_dict: dict
            Optimization parameters for the iteration controller of the
            transition optimization loop.
        point_key: str (default: 'point_sources')
            Key of the point source dofs in the MultiField of the joint
            reconstruction.
        stiffness: float (default: 1E6)
            Precision of the point source dof optimization after updating the
            diffuse components
        red_factor: float (default: 1E-3)
            Scaling for the convergence criterion regarding the second
            optimization for the point source dofs.
    """
    # TODO: replace second optimization with proper inverse transformation!
    def _transition(position):
        diffuse_pos = position.to_dict()
        diffuse_pos.pop(point_key)
        diffuse_pos = ift.MultiField.from_dict(diffuse_pos)

        my_sky = sky(position)

        lh = _IGLikelihood(my_sky, point_dict['alpha'], point_dict['q'])

        ic_mini = ift.AbsDeltaEnergyController(
                        deltaE = float(transition_dict['deltaE']),
                        iteration_limit = transition_dict['iteration_limit'],
                        convergence_level=transition_dict['convergence_level'],
                        name = transition_dict['name'])
        ham = ift.StandardHamiltonian(lh @ diffuse_sky)
        en, _ = ift.VL_BFGS(ic_mini)(ift.EnergyAdapter(diffuse_pos, ham))
        diffuse_pos = en.position

        new_pos = diffuse_pos.to_dict()
        new_pos['point_sources'] = position['point_sources']
        new_pos = ift.MultiField.from_dict(new_pos)

        icov = ift.ScalingOperator(my_sky.domain, stiffness)
        lh = ift.GaussianEnergy(data=my_sky, inverse_covariance=icov)
        en = ift.EnergyAdapter(new_pos, lh @ sky,
                               constants=list(diffuse_pos.keys()))
        ic_mini = ift.AbsDeltaEnergyController(
                        deltaE = red_factor * float(transition_dict['deltaE']),
                        iteration_limit = transition_dict['iteration_limit'],
                        convergence_level=transition_dict['convergence_level'],
                        name = transition_dict['name'])

        new_point_source_position = ift.VL_BFGS(ic_mini)(en)[0].position.to_dict()
        new_pos = new_pos.to_dict()
        new_pos['point_sources'] = new_point_source_position['point_sources']
        return ift.MultiField.from_dict(new_pos)

    _tr = (lambda samples: samples.average(_transition))
    return lambda iiter: None if iiter < transition_dict['start'] else _tr


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


# TODO do we have the equivalent?
def get_rel_uncertainty(mean, std):
    """Calculates the pointwise relative uncertainty from the mean
    and the standard deviation.

    Parameters
    ----------
    mean: nifty8.Field
    std: nifty8.Field

    Returns
    -------
    nifty8.Field
    """
    assert mean.domain == std.domain
    domain = mean.domain
    mean, std = mean.val, std.val
    res = np.zeros(mean.shape)
    mask = mean != 0
    res[mask] = std[mask] / mean[mask]
    res[~mask] = np.nan
    return ift.makeField(domain, res)


def get_RGB_image_from_field(field, norm=None, sat=None):
    """Turns a 3D Field into RGB image.

    Parameters
    ---------
    field: nifty8.Field
    norm: list
        containing normalization functions. Default:[np.log, np.log10, np.log10]
    sat: float
        Multiplicative factor defining maximum values in the RGB image.
        Default: None.

    Returns
    -------
    numpy.array
    """
    if norm is None:
        norm = [np.log, np.log10, np.log10]
    arr = field.val
    res = []
    for i in range(3):
        sub_array = arr[:, :, i]
        color_norm = norm[i]
        r = np.zeros_like(sub_array)
        mask = sub_array != 0
        if norm is not None:
            r[mask] = color_norm(sub_array[mask]) if color_norm is not None else sub_array[mask]
        min = np.min(r[mask])
        max = np.max(r[mask])
        r[mask] -= min
        r[mask] /= (max - min)
        if sat is not None:
            r[mask] *= sat[i]  # FIXME: this is not really saturation
        r[mask] = r[mask] * 255.0
        r[~mask] = 0
        res.append(r)
    res = np.array(res, dtype='int')
    res = np.transpose(res, (1, 2, 0))
    return res


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


def safe_config_update(key: str,
                       new_value,
                       config: dict,
                       verbose: bool = True) -> dict:
    """
    Update the configuration dictionary with a new value for the given key
    if the key is not already set or its current value is None.

    This function checks if the specified key exists in the given configuration
    dictionary and whether it has a non-None value. If the key is not present
    or its value is None, the function updates the dictionary with the provided
    new value. If both the existing value and new value are None, the function
    raises a ValueError.

    Parameters:
    -----------
    key : str
        The key to update in the configuration.
    new_value :
        The new value to set for the key.
    config : dict
        The configuration dictionary to update.
    verbose : bool, optional
        If True, print out the updated value. Default is True.

    Returns:
    --------
    dict
        The updated configuration dictionary.

    Raises:
    -------
    ValueError
        If the key is not set in the config and the new value is None.
    """
    if key in config and config[key] is not None:
        return config
    if new_value is not None:
        config[key] = new_value
        if verbose:
            print(f"{key} set to {new_value}.")
        return config
    raise ValueError(f"Either '{key}' must be set in the config file "
                     f"or a new value must be provided!")


# TODO do we need this -> to likelihood?
def get_n_constrained_dof(likelihood: jft.Likelihood) -> int:
    """
    Extacts the number of constrained degrees of freedom (DOF)
    based on the likelihood.

    Parameters
    ----------
    likelihood : jft.Likelihood
        The likelihood object which contains information about
       the model and data.

    Returns
    -------
    int
        The number of constrained degrees of freedom, which is the
        minimum of the model degrees of freedom and the data
        degrees of freedom.
    """

    n_dof_data = jft.size(likelihood.left_sqrt_metric_tangents_shape)
    n_dof_model = jft.size(likelihood.domain)
    return min(n_dof_model, n_dof_data)
