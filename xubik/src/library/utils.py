import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import nifty8 as ift


def get_cfg(yaml_file):
    import yaml
    with open(yaml_file, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg

def _get_sp_dist(config):
    res = 2 * config["fov"] / config["npix_s"]
    return res

def _get_e_dist(config):
    res = np.log(config["elim"][1] / config["elim"][0]) / config["npix_e"]
    return res

def get_data_domain(config):
    dom_sp = ift.RGSpace(([config["npix_s"]]*2), distances = _get_sp_dist(config))
    e_sp = ift.RGSpace((config["npix_e"]), distances = _get_e_dist(config))
    return ift.DomainTuple.make([dom_sp, e_sp])

def get_normed_exposure(exposure_field, data_field):
    dom = exposure_field.domain
    ratio = (
        data_field.val[exposure_field.val != 0]
        / exposure_field.val[exposure_field.val != 0]
    )
    norm = ratio.mean()
    normed_exp_field = exposure_field * norm
    return normed_exp_field

def get_norm_exposure_patches(datasets, domain, energy_bins):
    norms = []
    norm_mean = []
    norm_max = []
    norm_std = []
    for i in range(energy_bins):
        for dataset in datasets:
            observation = np.load(dataset, allow_pickle=True).item()
            exposure = observation["exposure"].val[:,:,i]
            data = observation["data"].val[:,:,i]
            norms.append(get_norm(ift.Field.from_raw(domain, exposure), ift.Field.from_raw(domain, data)))
        norm_mean.append(np.mean(np.array(norms)))
        norm_max.append(np.amax(np.array(norms)))
        norm_std.append(np.std(np.array(norms)))
    return norm_max, norm_mean, norm_std

def get_norm(exposure_field, data_field):
    dom = exposure_field.domain
    ratio = (
        data_field.val[exposure_field.val != 0]
        / exposure_field.val[exposure_field.val != 0]
    )
    norm = ratio.mean()
    # norm = 10**math.floor(math.log10(norm))
    return norm

def prior_sample_plotter(opchain, n):
    fig, ax = plt.subplots(1, n, figsize=(11.7, 8.3), dpi=200)
    ax = ax.flatten()
    for ii in range(n):
        f = ift.from_random(opchain.domain)
        field = opchain(f)
        fov = (
            field.domain[0].distances[0] * field.domain[0].shape[0] / 2.0
        )  # is this true?
        pltargs = {
            "origin": "lower",
            "cmap": "inferno",
            "extent": [-fov, fov] * 2,
            "norm": LogNorm(),
        }
        img = field.val
        im = ax[ii].imshow(img, **pltargs)
        cb = fig.colorbar(im, ax=ax[ii])
    fig.tight_layout()
    plt.show()
    plt.close()


def get_mask_operator(exp_field):
    mask = np.zeros(exp_field.shape)
    mask[exp_field.val == 0] = 1
    mask_field = ift.Field.from_raw(exp_field.domain, mask)
    mask_operator = ift.MaskOperator(mask_field)
    return mask_operator


# FIXME actually here are pixels (Bad Pixels?) in the middle of the data which are kind of dead which are NOT included in the expfield
# this should be fixed, otherwise we could run into problems with the reconstruction


def convolve_operators(a, b):
    FFT = ift.FFTOperator(a.target)
    convolved = FFT.inverse(FFT(a.real) * FFT(b.real))
    return convolved.real


def convolve_field_operator(kernel, op, space=None):
    op = op.real
    fft = ift.FFTOperator(op.target, space=space)
    hsp_kernel = fft(kernel.real)
    kernel_hp = ift.makeOp(hsp_kernel)
    convolve = fft.inverse @ kernel_hp @ fft @ op
    res = convolve.real
    return res
#FIXME Hartley + Fix dirty hack

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
    This Operator performs a coordinate transformation into a coordinate system,
    in which the Oth component is the sum of all components of the former basis.
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

def save_rgb_image_to_fits(fld, file_name, overwrite, mpi):
    import astropy.io.fits as pyfits
    from astropy.time import Time
    import time
    color_dict = {0: "red", 1: "green", 2: "blue"}
    domain = fld.domain
    if not isinstance(domain, ift.DomainTuple) or len(domain)!=2 or len(domain[0].shape)!=2:
        raise ValueError("FITS file export of RGB data is only possible for 3d-fields. "
                         f"Current domain:\n{domain}")
    if fld.shape[2] != 3:
        raise ValueError("Energy direction has to be binned to 3 to create an RGB image. "
                         f"Current number of energy bins:\n{fld.shape[2]}")
    h = pyfits.Header()
    h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]
    h["CRVAL1"] = h["CRVAL2"] = 0
    h["CRPIX1"] = h["CRPIX2"] = 0
    h["CUNIT1"] = h["CUNIT2"] = "deg"
    h["CDELT1"], h["CDELT2"] = -domain[0].distances[0], domain[0].distances[1]
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC---SIN"
    h["EQUINOX"] = 2000
    if mpi:
        for i in range(fld.shape[2]):
            hdu = pyfits.PrimaryHDU(fld.val[:,:,i], header=h)
            hdulist = pyfits.HDUList([hdu])
            file_name_colour = f"{file_name}_{color_dict[i]}.fits"
            hdulist.writeto(file_name_colour, overwrite=overwrite)

def energy_binning(fld, energy_bins, data_type=None):
    domain = fld.domain
    arr = fld.val
    shape = [i for i in arr.shape]
    new_shape = shape[:2]
    new_shape.append(energy_bins)
    new_domain = ift.DomainTuple.make((domain[0], ift.RGSpace(energy_bins)))
    aux_arrs =[]
    binned_array = arr
    if shape[2]<energy_bins:
        binned_array = np.pad(arr, [(0,0), (0,0), (0,(energy_bins-shape[2]))], mode='constant')
    if shape[2]>energy_bins:
        bins = np.arange(0, shape[2]+1, shape[2]/(energy_bins))
        for i in range(len(bins)-1):
            bin1 = int(bins[i])
            bin2 = int(bins[i+1])
            aux_arrs.append(np.sum(arr[:,:,bin1:bin2], axis=2))
        binned_array = np.stack(aux_arrs, axis=2)
    binned_field = ift.Field.from_raw(new_domain, binned_array)
    return binned_field

def transform_loglog_slope_pars(slope_pars):
    """Transform slope parameters from log10/log10 to ln/log10 space"""
    res = slope_pars.copy()
    res['mean'] = (res['mean']+1) *np.log(10)
    res['sigma'] *=np.log(10)
    return res
