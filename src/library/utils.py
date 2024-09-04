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
