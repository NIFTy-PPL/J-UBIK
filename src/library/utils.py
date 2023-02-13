import os
from warnings import warn
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

import nifty8 as ift


def get_cfg(yaml_file):
    """
    Convenience function for loading yaml-config files
    """
    import yaml
    with open(yaml_file, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg


def save_config(config, filename, dir=None):
    import yaml
    if dir is not None:
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(os.path.join(dir, filename), "w") as f:
            yaml.dump(config, f)


def create_output_directory(directory_name):
    os.makedirs(directory_name, exist_ok=True)
    return directory_name


def get_gaussian_psf(op, var):
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


def get_data_domain(config):
    dom_sp = ift.RGSpace(([config["npix_s"]] * 2), distances=_get_sp_dist(config))
    e_sp = ift.RGSpace((config["npix_e"]), distances=_get_e_dist(config))
    return ift.DomainTuple.make([dom_sp, e_sp])


def _get_sp_dist(config):
    res = config["fov"] / config["npix_s"]
    return res


def _get_e_dist(config):
    res = np.log(config["elim"][1] / config["elim"][0]) / config["npix_e"]
    return res

def get_normed_exposure(exposure_field, data_field):
    """
    Convenience function to get exposures on the order of 1, so that the signal is living on
    the same order of magnitude as the data.
    """
    warn("get_normed_exposure: This feauture was used for development only and will be deprecated soon.", DeprecationWarning, stacklevel=2)
    ratio = (
            data_field.val[exposure_field.val != 0]
            / exposure_field.val[exposure_field.val != 0]
    )
    norm = ratio.mean()
    normed_exp_field = exposure_field * norm
    return normed_exp_field

def get_norm_exposure_patches(datasets, domain, energy_bins, obs_type=None):
    warn("get_norm_exposure_patches: This feauture was used for development only and will be deprecated soon.", DeprecationWarning, stacklevel=2)
    norms = []
    norm_mean = []
    norm_max = []
    norm_std = []
    if obs_type == None:
        obs_type = 'SF'
    for i in range(energy_bins):
        for dataset in datasets:
            observation = np.load("npdata/" + obs_type + "/df_" + str(dataset) + "_observation.npy",
                                  allow_pickle=True).item()
            exposure = observation["exposure"].val[:, :, i]
            data = observation["data"].val[:, :, i]
            norms.append(get_norm(ift.Field.from_raw(domain, exposure), ift.Field.from_raw(domain, data)))
        norm_mean.append(np.mean(np.array(norms)))
        norm_max.append(np.amax(np.array(norms)))
        norm_std.append(np.std(np.array(norms)))
    return norm_max, norm_mean, norm_std


def get_norm(exposure_field, data_field):
    """
    returns only the order of magnitude of
    the norm of get_normed_exposure
    """
    warn("get_norm: This feauture was used for development only and will be deprecated soon.", DeprecationWarning, stacklevel=2)
    ratio = (
            data_field.val[exposure_field.val != 0]
            / exposure_field.val[exposure_field.val != 0]
    )
    norm = ratio.mean()
    # norm = 10**math.floor(math.log10(norm))
    return norm


def get_mask_operator(exp_field):
    mask = np.zeros(exp_field.shape)
    mask[exp_field.val == 0] = 1
    mask_field = ift.Field.from_raw(exp_field.domain, mask)
    mask_operator = ift.MaskOperator(mask_field)
    return mask_operator

    # FIXME actually here are pixels (Bad Pixels?) in the middle of
    # the data, which are kind of dead which are NOT included in the
    # expfield this should be fixed, otherwise we could run into
    # problems with the reconstruction


def prior_sample_plotter(opchain, n):
    """
    Convenience function for prior sample plotting.
    #TODO Check if this is in nifty8 --> than this can be deleted
    """
    fig, ax = plt.subplots(1, n, figsize=(11.7, 8.3), dpi=200)
    ax = ax.flatten()
    for ii in range(n):
        f = ift.from_random(opchain.domain)
        field = opchain(f)
        half_fov = (
                field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0
        )  # is this true?
        pltargs = {
            "origin": "lower",
            "cmap": "inferno",
            "extent": [-half_fov, half_fov] * 2,
            "norm": LogNorm(),
        }
        img = field.val
        im = ax[ii].imshow(img, **pltargs)
        cb = fig.colorbar(im, ax=ax[ii])
    fig.tight_layout()
    plt.show()
    plt.close()


def get_psfpatches(info, n, npix_s, ebin, fov, num_rays=10e6, debug=False, Roll=True, Norm=True):
    psf_domain = ift.RGSpace((npix_s, npix_s), distances=fov / npix_s)
    xy_range = info.obsInfo["xy_range"]
    x_min = info.obsInfo["x_min"]
    y_min = info.obsInfo["y_min"]
    dy = dx = xy_range * 2 / n
    x_i = x_min + dx * 1 / 2
    y_i = y_min + dy * 1 / 2
    coords = coord_center(npix_s, n)
    psf_sim = []
    source = []
    u = 0
    positions = []
    for i in range(n):
        for l in range(n):
            x_p = x_i + i * dx
            y_p = y_i + l * dy
            radec_c = get_radec_from_xy(x_p, y_p, info.obsInfo["event_file"])
            tmp_psf_sim = info.get_psf_fromsim(radec_c, outroot="./psf", num_rays=num_rays)
            tmp_psf_sim = tmp_psf_sim[:, :, ebin]
            if Roll:
                tmp_coord = coords[u]
                co_x, co_y = np.unravel_index(tmp_coord, [npix_s, npix_s])
                tmp_psf_sim = np.roll(tmp_psf_sim, (-co_x, -co_y), axis=(0, 1))
                u += 1
            psf_field = ift.makeField(psf_domain, tmp_psf_sim)
            if Norm:
                norm_val = psf_field.integrate().val ** -1
                norm = ift.ScalingOperator(psf_domain, norm_val)
                psf_norm = norm(psf_field)
                psf_sim.append(psf_norm)
            else:
                psf_sim.append(psf_field)

            if debug:
                tmp_source = np.zeros(tmp_psf_sim.shape)
                pos = np.unravel_index(np.argmax(tmp_psf_sim, axis=None), tmp_psf_sim.shape)
                tmp_source[pos] = 1
                source_field = ift.makeField(psf_domain, tmp_source)
                source.append(source_field)
                positions.append(pos)
    if debug:
        return psf_sim, source, positions, coords
    else:
        return psf_sim


def get_synth_pointsource(info, npix_s, fov, idx_tupel, num_rays):
    xy_range = info.obsInfo["xy_range"]
    x_min = info.obsInfo["x_min"]
    y_min = info.obsInfo["y_min"]
    event_f = info.obsInfo["event_file"]
    dy = dx = xy_range * 2 / npix_s
    x_idx, y_idx = idx_tupel
    x_pix_coord = x_min + x_idx * dx
    y_pix_coord = y_min + y_idx * dy
    coords = get_radec_from_xy(x_pix_coord, y_pix_coord, event_f)
    ps = info.get_psf_fromsim(coords, outroot="./psf", num_rays=num_rays)
    return ps


def coord_center(side_length, side_n):
    """
    calculates the indices of the centers of the n**2 patches
    for a quadratical domain with a certain side length

    Parameters:
    ----------
    side_length: int
        length of one side
    side_n: int
        number of patches along one side
    """
    tdx = tdy = side_length // side_n
    xc = np.arange(tdx // 2, tdx * side_n, tdx)
    yc = np.arange(tdy // 2, tdy * side_n, tdy)
    co = np.array(np.meshgrid(xc, yc)).reshape(2, -1)
    res = np.ravel_multi_index(co, [side_length, side_length])
    return res


def get_radec_from_xy(temp_x, temp_y, event_f):
    import ciao_contrib.runtool as rt

    rt.dmcoords.punlearn()
    rt.dmcoords(event_f, op="sky", celfmt="deg", x=temp_x, y=temp_y)
    x_p = float(rt.dmcoords.ra)
    y_p = float(rt.dmcoords.dec)
    return (x_p, y_p)
    # TODO is this enough precision


def convolve_operators(a, b):
    """
    convenience function for the convolution of two operators a and b.
    This uses Fast Fourier Transformation (FFT).
    """
    FFT = ift.FFTOperator(a.target)
    convolved = FFT.inverse(FFT(a.real) * FFT(b.real))
    return convolved.real


def convolve_field_operator(kernel, op, space=None):
    """
    convenience function for the convolution a fixed kernel (field) with an operator.
    This uses Fast Fourier Transformation (FFT).
    """
    convolve_op = get_fft_psf_op(kernel, op.target, space)
    return convolve_op @ op


def get_fft_psf_op(kernel, domain, space=None):
    """
    convenience function for the generation of a convolution operator with fixed kernel (field).
    This uses Fast Fourier Transformation (FFT).
    """
    fft = ift.FFTOperator(domain, space=space)
    realizer = ift.Realizer(domain)
    hsp_kernel = fft(kernel.real)
    kernel_hp = ift.makeOp(hsp_kernel)
    return realizer @ fft.inverse @ kernel_hp @ fft @ realizer
    # FIXME Hartley + Fix dirty hack


class PositiveSumPriorOperator(ift.LinearOperator):
    """
    Operator performing a coordinate transformation, requiring MultiToTuple and Trafo.
    """

    def __init__(self, domain, target=None):
        self._domain = domain
        if not isinstance(self._domain, ift.MultiDomain):
            raise TypeError("domain must be a MultiDomain")
        if target == None:
            self._target = self._domain
        else:
            self._target = target
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._multi = MultiToTuple(self._domain)
        self._trafo = Trafo(self._multi.target)

    def apply(self, x, mode):
        self._check_input(x, mode)
        op = self._multi.adjoint @ self._trafo @ self._multi
        if mode == self.TIMES:
            res = op(x)
        else:
            res = op.adjoint(x)
        return res


class MultiToTuple(ift.LinearOperator):
    """
    Puts several Fields of a Multifield of the same domains, into a Domaintuple
    along a UnstructuredDomain.
    """

    def __init__(self, domain):
        self._domain = domain
        if not isinstance(self._domain, ift.MultiDomain):
            raise TypeError("domain has to be a ift.MultiDomain")
        self._first_dom = domain[domain.keys()[0]][0]
        for key in self._domain.keys():
            if not self._first_dom == domain[key][0]:
                raise TypeError("All sub domains must be equal ")
        n_doms = ift.UnstructuredDomain(len(domain.keys()))
        self._target = ift.makeDomain((n_doms, self._first_dom))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            lst = []
            for key in x.keys():
                lst.append(x[key].val)
            x = np.array(lst)
            res = ift.Field.from_raw(self._target, x)
        else:
            dct = {}
            ii = 0
            for key in self._domain.keys():
                tmp_field = ift.Field.from_raw(self._first_dom, x.val[ii, :, :])
                dct.update({key: tmp_field})
                ii += 1
            res = ift.MultiField.from_dict(dct)
        return res


class Trafo(ift.EndomorphicOperator):
    """
    #NOTE RENAME TRAFO
    This Operator performs a coordinate transformation into a coordinate
    system, in which the Oth component is the sum of all components of
    the former basis.
    """

    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._n = self.domain.shape[0]
        self._build_mat()
        self._capability = self.TIMES | self.ADJOINT_TIMES
        lamb, s = self._build_mat()
        self._lamb = lamb
        if not np.isclose(lamb[0], 0):
            raise ValueError(
                "Transformation does not work, check eigenvalues self._lamb"
            )
        self._s = s
        if s[0, 0] < 0:
            s[:, 0] = -1 * s[:, 0]
        self._s_inv = scipy.linalg.inv(s)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            y = np.einsum("ij, jmn->imn", self._s, x)
        else:
            y = np.einsum("ij, jmn-> imn", self._s_inv, x)
        return ift.Field.from_raw(self._tgt(mode), y)

    def _build_mat(self):
        l = self._n
        one = np.zeros([l] * 2)
        np.fill_diagonal(one, 1)

        norm_d = np.ones([l] * 2) / l
        proj = one - norm_d
        eigv, s = np.linalg.eigh(proj)
        return eigv, s


def get_distributions_for_positive_sum_prior(domain, number):
    for i in range(number):
        field_adapter = ift.FieldAdapter(domain, f"amp_{i}")
        tmp_operator = field_adapter.adjoint @ field_adapter
        if i == 0:
            operator = tmp_operator.exp()
        else:
            operator = operator + tmp_operator
    return operator


def makePositiveSumPrior(domain, number):
    distributions = get_distributions_for_positive_sum_prior(domain, number)
    positive_sum = PositiveSumPriorOperator(distributions.target)
    op = positive_sum @ distributions
    return op


def field_T(field):
    """
    Getting the transposed field of the original field.
    This only works for quadratical domains.
    """
    domain = field.domain
    arr = field.val.T
    res = ift.Field.from_raw(domain, arr)
    return res


class Transposer(ift.EndomorphicOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = self.domain
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = ift.Field.from_raw(self._tgt(mode), x.val.T)
        return res


def save_to_fits(sample_list, file_name_base, op=None, samples=False, mean=False, std=False,
                 overwrite=False, obs_type="SF"):
    """Write sample list to FITS file.

    This function writes properties of a sample list to a FITS file according to the obs_type and based on the NIFTy8
    function save_to_fits by P.Arras

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
        Describes the observation type. currently possible obs_types are [CMF (Chandra Multifrequency),
        EMF (Erosita Multifrequency), RGB and SF (Single Frequency]. The default observation is of type SF. In the case
        of the type "RGB", the binning is automatically done by xubik into equally sized bins.
    """
    if not (samples or mean or std):
        raise ValueError("Neither samples nor mean nor standard deviation shall be written.")

    if mean or std:
        m, s = sample_list.sample_stat(op)

    if obs_type == "RGB":
        m = energy_binning(m, energy_bins=3)
        s = energy_binning(s, energy_bins=3)
    if mean:
        save_rgb_image_to_fits(m, file_name_base + "_mean", overwrite, sample_list.MPI_master)
    if std:
        save_rgb_image_to_fits(s, file_name_base + "_std", overwrite, sample_list.MPI_master)
    if samples:
        for ii, ss in enumerate(sample_list.iterator(op)):
            if obs_type == "RGB":
                ss = energy_binning(ss, energy_bins=3)
            save_rgb_image_to_fits(ss, file_name_base + f"_sample_{ii}", overwrite, sample_list.MPI_master)


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


def energy_binning(fld, energy_bins):
    """
    Takes a field with an arbitrary number of energy bins and reshapes it into a field with three energy-bins.
    Parameters. If the field has less than 3 energy-bins the field is padded with a constant value. If the field
    has 3 energy bins, nothing happens and if the field has more than 3 energy bins the array is rebinned to three
    equally sized energy bins.

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
    If the number of energy-bins divided by 3 is not an integer, the last bin will be bigger.
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


def transform_loglog_slope_pars(slope_pars):
    """Transform slope parameters from log10/log10 to ln/log10 space"""
    res = slope_pars.copy()
    res['mean'] = (res['mean'] + 1) * np.log(10)
    res['sigma'] *= np.log(10)
    return res


def is_subdomain(sub_domain, total_domain):
    if not isinstance(sub_domain, (ift.MultiDomain, ift.DomainTuple)):
        raise TypeError
    if isinstance(sub_domain, ift.DomainTuple):
        return sub_domain == total_domain
    return all(kk in total_domain.keys() and vv == total_domain[kk]
               for kk, vv in sub_domain.items())


def get_data_realization(op, position, exposure=None, padder=None, data=True, output_directory=None):
    mpi_master = ift.utilities.get_MPI_params()[3]
    R = ift.ScalingOperator(op.target, 1)
    if exposure is not None:
        R = exposure @ R
    R_no_pad = R

    if padder is not None:
        R = padder.adjoint @ R
    res = op.force(position)
    if data:
        res = R(op.force(position))
        res = ift.random.current_rng().poisson(res.val.astype(np.float64))
        if padder is not None:
            res = ift.makeField(padder.adjoint.target, res)
        else:
            res = ift.makeField(op.target, res)
    if output_directory is not None and mpi_master:
        with open(os.path.join(output_directory, f'response_op.pkl'), 'wb') as file:
            pickle.dump(R_no_pad, file)
    return res


def generate_mock_data(sky_model, psf_op, exposure=None, pad=None, alpha=None, q=None, n=None,
                       output_directory=None):
    if pad is None and sky_model.position_space != sky_model.extended_space:
        raise ValueError('The sky is padded but no padder is given')
    mpi_master = ift.utilities.get_MPI_params()[3]

    # Create output and diagnostic directories
    if output_directory is not None:
        diagnostics_dir = os.path.join(output_directory, 'diagnostics')
        if mpi_master:
            create_output_directory(output_directory)
            create_output_directory(diagnostics_dir)

    # Exposure
    exposure_field = exposure
    if exposure is not None:
        if pad is not None:
            exposure = pad(exposure_field)
        exposure = ift.makeOp(exposure)

    # Get sky operators
    sky_dict = sky_model.create_sky_model()
    sky_dict.pop('pspec')

    # Mock sky
    mock_sky_position = ift.from_random(sky_dict['sky'].domain)
    mock_sky = sky_dict['sky'](mock_sky_position)
    mock_sky_data = get_data_realization(sky_dict['sky'], mock_sky_position, exposure=exposure, padder=pad,
                                         output_directory=diagnostics_dir)

    # Convolve sky operators and draw data realizations
    conv_sky_dict = {key: (psf_op @ value) for key, value in sky_dict.items()}
    prefix = "mock_data_"
    mock_data_dict = {prefix+key: get_data_realization(value,
                                                       mock_sky_position,
                                                       exposure=exposure,
                                                       padder=pad) for key, value in conv_sky_dict.items()}

    # Prepare output dictionary
    output_dictionary = {**mock_data_dict,
                         'mock_sky': mock_sky}

    if mpi_master and output_directory is not None:
        p = ift.Plot()
        for k, v in output_dictionary.items():
            # Save data and sky to Pickle
            with open(os.path.join(diagnostics_dir, f'{k}.pkl'), 'wb') as file:
                pickle.dump(v, file)

            # Save data to fits
            save_rgb_image_to_fits(v, os.path.join(diagnostics_dir, k), overwrite=True, MPI_master=mpi_master)

            # Plot data
            p.add(v, title=k, norm=LogNorm())
        if exposure_field is not None:
            p.add(exposure_field, title='exposure', norm=LogNorm())
        p.output(nx=3, name=os.path.join(diagnostics_dir, f'mock_data_a{alpha}_q{q}_sample{n}.png'))

    return mock_data_dict