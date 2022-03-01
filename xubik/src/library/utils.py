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

def get_norm(exposure_field, data_field):
    dom = exposure_field.domain
    ratio = (
        data_field.val[exposure_field.val != 0]
        / exposure_field.val[exposure_field.val != 0]
    )
    norm = ratio.mean()
    norm = 10**math.floor(math.log10(norm))
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


def coord_center(side_length, side_n):
    tdx = tdy = side_length // side_n
    xc = np.arange(tdx // 2, tdx * side_n, tdx // 2)
    yc = np.arange(tdy // 2, tdy * side_n, tdy // 2)
    co = np.array(np.meshgrid(xc, yc)).reshape(2, -1)
    # res = np.ravel_multi_index(co, [side_length, side_length])
    return co

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
