import nifty8 as ift
import xubik0 as xu
import numpy as np


def test_overlapadd():
    position_space = ift.RGSpace([1024, 1024])
    args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (1., 0.8),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2, 1.),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.4)  # 0.1, 0.5
    }
    correlated_field = ift.SimpleCorrelatedField(position_space, **args)
    xi = ift.from_random(correlated_field.domain)
    cf = correlated_field(xi)
    zp = xu.MarginZeroPadder(position_space, 128)
    zp_cf = zp(cf)
    margin = 64
    n = 64

    kern_domain = ift.makeDomain([ift.UnstructuredDomain(64), position_space])
    kernels_arr = xu.get_gaussian_kernel(200, kern_domain).val
    convolve_oa = xu.OverlapAddConvolver(zp.target, kernels_arr, n, margin)

    res_1 = convolve_oa(zp_cf)
    res_1 = zp.adjoint(res_1)

    kern = ift.Field.from_raw(position_space, kernels_arr[0])
    kern_zeropadder = ift.FieldZeroPadder(kern.domain, [1280, 1280], central=True)
    kern = kern_zeropadder(kern)
    sig_2 = zp @ correlated_field
    response_2 = xu.convolve_field_operator(kern, sig_2)
    response_2 = zp.adjoint @ response_2
    res_2 = response_2(xi)
    print(np.allclose(res_2.val, res_1.val))
    pl = ift.Plot()
    pl.add(cf, title="signal")
    pl.add(res_1, title="conv with overlapadd")
    pl.add(res_2, title="conv theorem convolution")
    pl.add(res_1-res_2, title="difference")
    pl.output(name="test_convolution.png")
