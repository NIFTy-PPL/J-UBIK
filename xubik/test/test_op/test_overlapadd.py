import nifty8 as ift
import xubik0 as xu
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LogNorm

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
    margin = 160
    n = 64

    kern_domain = ift.makeDomain([ift.UnstructuredDomain(64), position_space])
    kernels_arr = xu.get_gaussian_kernel(35, kern_domain).val_rw()
    # convolve_oa = xu.OverlapAddConvolver(zp.target, kernels_arr, n, margin)

    convolve_oa = xu.OAConvolver.force(cf.domain, kernels_arr, n, margin)
    res_1 = convolve_oa(cf)

    # res_1 = convolve_oa(zp_cf)
    # res_1 = zp.adjoint(res_1)

    kern = ift.Field.from_raw(position_space, kernels_arr[0])
    kern_zeropadder = ift.FieldZeroPadder(kern.domain, [1280, 1280], central=True)
    kern = kern_zeropadder(kern)
    sig_2 = zp @ correlated_field
    response_2 = xu.convolve_field_operator(kern, sig_2)
    response_2 = zp.adjoint @ response_2
    res_2 = response_2(xi)

    f1 = ift.Field.full(convolve_oa.domain, 1.2)
    # f1_s = zp.adjoint(f1)
    # res_3 = zp.adjoint(convolve_oa(f1))

    psf_file = np.load("../../../data/npdata/psf_patches/obs4952_patches_fov15.npy", allow_pickle=True).item()["psf_sim"]
    # psf_file = len(psf_file) * [psf_file[0]]
    psfs = []
    for p in psf_file:
        arr = p.val_rw()
        # arr[arr < 5] = 0
        # arr[arr >= 0] = 1
        # arr[0, 0] = 1
        # p = ift.Field.from_raw(position_space, arr)
        # p = p/p.s_integrate()
        # psfs.append(p.val)
        psfs.append(arr)
    psfs = np.array(psfs, dtype="float64")
    # convolve_oa_ch = xu.OverlapAddConvolver(zp.target, psfs, n, margin)

    # res_4 = zp.adjoint(convolve_oa_ch(f1))
    # s_ps = ift.RGSpace([896, 896], distances= position_space.distances)
    # cut = xu.MarginZeroPadder(s_ps, 64).adjoint
    # res_4 = cut(res_4)

    oa2 = xu.OAConvolver.force(position_space, psfs, n, margin)
    f2 = ift.Field.full(position_space, 1.2)
    res_5 = oa2(f2)
    res_6 = oa2(cf)
    zp3 = xu.MarginZeroPadder(ift.RGSpace([768, 768], distances=res_5.domain[0].distances), 128)
    res_5 = zp3.adjoint(res_5)
    res_6 = zp3.adjoint(res_6)
    print(np.min(res_5.val), np.max(res_5.val))

    pl = ift.Plot()
    pl.add(cf, title="signal")
    pl.add(res_1, title="conv with overlapadd")
    pl.add(res_2, title="conv theorem convolution")
    pl.add(zp3.adjoint(ift.abs(res_1-res_2)), title="difference")
    # pl.add(f1_s, title="ones")
    pl.output(name="test_convolution.png")
    pl1 = ift.Plot()
    # pl1.add(res_3, norm=LogNorm(), title="conv ones with gauss")
    # pl1.add(res_4, norm=LogNorm(), title="conv ones with chandra kernel")
    pl1.add(kern+1e-10, norm=SymLogNorm(1), title="gauss kernel")
    # pl1.add(ift.Field.from_raw(p.domain, psfs[0]), norm=SymLogNorm(1), title="chandra")
    # print(np.min(res_3.val), np.max(res_3.val))
    # print(np.min(res_4.val), np.max(res_4.val))
    pl1.add(res_5, title="conv ones with chandra kernel")
    pl1.add(res_6, title="conv cf with chandra kernel")
    pl1.add(cf, title="cf")
    pl1.output(name="test2.png")

    # print(np.unique(psfs[0]))
    # ift.extra.assert_equal(psf_file2[28], psf_file[28])
    # ift.extra.assert_allclose(res_2, res_1)
    # ift.extra.assert_allclose(f1_s, res_3)
