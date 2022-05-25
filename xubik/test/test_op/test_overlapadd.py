import nifty8 as ift
import xubik0 as xu
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LogNorm

def test_oaconvolver():
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
    margin = 128
    n = 64

    # Convolve Correlated Field with Gauss Kernel via Interpolation
    kern_domain = ift.makeDomain([ift.UnstructuredDomain(64), position_space])
    kernels_arr = xu.get_gaussian_kernel(35, kern_domain).val_rw()
    convolve_oa = xu.OAConvolver.force(cf.domain, kernels_arr, n, margin)
    res_1 = convolve_oa(cf)

    # Convolve Correlated Field via Conv Theorem
    # Kernel preparation and padding
    kern = ift.Field.from_raw(position_space, kernels_arr[0])
    kern_zeropadder = ift.FieldZeroPadder(kern.domain, [1280, 1280], central=True)
    kern = kern_zeropadder(kern)
    # signal padding
    zp = xu.MarginZeroPadder(position_space, margin)
    sig_2 = zp @ correlated_field  # Break PBC

    # Convolve and remove PBC Margin
    response_2 = zp.adjoint @ xu.convolve_field_operator(kern, sig_2)

    # Cut interpolation Margin as in res_1, just for comparision
    inner_shape = [i - 2*margin for i in position_space.shape]
    inner_domain = ift.RGSpace(inner_shape, distances=position_space.distances)
    cut_interpolate_margin = xu.MarginZeroPadder(inner_domain, margin).adjoint
    res_2 = (cut_interpolate_margin @ response_2)(xi)

    # Check if test holds
    diff = ift.abs(res_1 - res_2)
    np.testing.assert_allclose(res_1.val, res_2.val)

    # Plotting
    pl = ift.Plot()
    pl.add(cf, title="signal")
    pl.add(res_1, title="conv with overlapadd")
    pl.add(res_2, title="conv theorem convolution")
    pl.add(diff, title="difference")
    pl.output(name="test_convolution.png")

    # Test Normalization with spatially invariant gauss kernel
    f1 = ift.Field.full(position_space, 1.2)
    f1_s = cut_interpolate_margin(f1)
    res_3 = convolve_oa(f1)
    np.testing.assert_allclose(res_3.val, f1_s.val)


    pl = ift.Plot()
    pl.add(f1_s, title="full ones")
    pl.add(res_3, title="conv ones with gaussian")
    pl.output(name="test_normalization_gauss.png")

    # Load Chandra Kernel
    psf_file = np.load("../../../data/npdata/psf_patches/obs4952_patches_v1.npy", allow_pickle=True).item()["psf_sim"]
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

    # Test inhomogenous psf
    oa_chandra = xu.OAConvolver.force(position_space, psfs, n, margin)
    oa_chandra = xu.OAConvolver.cut_by_value(position_space, psfs, n, margin, 50)
    res_4 = oa_chandra(cf)

    pl = ift.Plot()
    pl.add(cf, title="Correlated Field Signal")
    pl.add(res_4, title="Inhomogeneous PSF Response")
    pl.output(name="test_inhomogeneous.png")

    # Test Normalization of inhomogeneous psf
    f2 = ift.Field.full(position_space, 1.2)
    f2_s = cut_interpolate_margin(f2)
    res_5 = oa_chandra(f2)
    np.testing.assert_allclose(res_5.val, f2_s.val)

    pl = ift.Plot()
    pl.add(f2_s, title="full ones")
    pl.add(res_5, title="conv ones with inhomogeneous psf of chandra")
    pl.output(name="test_normalization_chandra.png")

    # Test Conservation of Flux
    # FIXME This is somewhat strange
    mock_field = np.zeros([1024, 1024])
    # mock_field[300,300] = 30000
    mock_field[300, 300] = 1
    # mock_field[700,700] = 30000
    mock_field = ift.Field.from_raw(position_space, mock_field)

    cut_mock_field = cut_interpolate_margin(mock_field)
    oa_mock_field = oa_chandra(mock_field)
    test = cut_mock_field-oa_mock_field
    res_6 = cut_mock_field.integrate()
    res_7 = oa_mock_field.integrate()

    mock_op = zp @ ift.ScalingOperator(position_space, 1)
    full_op = zp.adjoint(xu.convolve_field_operator(kern, mock_op))
    conv_fft = full_op(mock_field)
    res_8 = cut_interpolate_margin(conv_fft).integrate()

    conv_oa_gauss = convolve_oa(mock_field)
    res_9 = conv_oa_gauss.integrate()

    np.testing.assert_allclose(res_6.val, res_7.val)
    pl = ift.Plot()
    pl.add(cut_mock_field, title="Some Points")
    pl.add(oa_mock_field, title="Convolved with OAChandra")
    pl.add(conv_fft, title="Convolved with FFT")
    pl.add(conv_oa_gauss, title="Convolved with OA Gaussian ")
    pl.output(name="test_conservation.png")


def test_oanew():
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
    margin = 128
    n = 64

    # Convolve Correlated Field with Gauss Kernel via Interpolation
    kern_domain = ift.makeDomain([ift.UnstructuredDomain(64), position_space])
    kernels_arr = xu.get_gaussian_kernel(35, kern_domain).val_rw()
    convolve_oa = xu.OAnew.force(cf.domain, kernels_arr, n, margin)
    # ift.extra.check_linear_operator(convolve_oa)
    res_1 = convolve_oa(cf)

    # Convolve Correlated Field via Conv Theorem
    # Kernel preparation and padding
    kern = ift.Field.from_raw(position_space, kernels_arr[0])
    kern_zeropadder = ift.FieldZeroPadder(kern.domain, [1024+2*margin]*2, central=True)
    kern = kern_zeropadder(kern)
    # signal padding
    zp = xu.MarginZeroPadder(position_space, margin)
    sig_2 = zp @ correlated_field  # Break PBC

    # Convolve and remove PBC Margin
    response_2 = zp.adjoint @ xu.convolve_field_operator(kern, sig_2)

    # Cut interpolation Margin as in res_1, just for comparision
    interpolation_margin = int(position_space.shape[0]//np.sqrt(n)*2)
    inner_shape = [i - 2*interpolation_margin for i in position_space.shape]
    inner_domain = ift.RGSpace(inner_shape, distances=position_space.distances)
    cut_interpolate_margin = xu.MarginZeroPadder(inner_domain, interpolation_margin).adjoint
    res_2 = (cut_interpolate_margin @ response_2)(xi)

    # Check if test holds
    diff = ift.abs(res_1 - res_2)
    # np.testing.assert_allclose(res_1.val, res_2.val)

    # Plotting
    pl = ift.Plot()
    pl.add(cf, title="signal")
    pl.add(res_1, title="conv with overlapadd")
    pl.add(res_2, title="conv theorem convolution")
    pl.add(diff, norm=LogNorm(), title="difference")
    pl.output(name="test_convolution_2.png")

    # Test Normalization with spatially invariant gauss kernel
    f1 = ift.Field.full(position_space, 1.2)
    f1_s = cut_interpolate_margin(f1)
    res_3 = convolve_oa(f1)
    # np.testing.assert_allclose(res_3.val, f1_s.val)


    pl = ift.Plot()
    pl.add(f1_s, title="full ones")
    pl.add(res_3, title="conv ones with gaussian")
    pl.output(name="test_normalization_gauss_2.png")

    # Load Chandra Kernel
    psf_file = np.load("../../../data/npdata/psf_patches/obs4952_patches_v1.npy", allow_pickle=True).item()["psf_sim"]
    # psf_file = np.load("../../../data/npdata/psf_patches/shifted.npy", allow_pickle=True).item()["psf_sim"]
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

    # Test inhomogenous psf
    oa_chandra = xu.OAnew.cut_by_value(position_space, psfs, n, margin, 6)
    # ift.extra.check_linear_operator(oa_chandra)
    res_4 = oa_chandra(cf)

    pl = ift.Plot()
    pl.add(cf, title="Correlated Field Signal")
    pl.add(ift.abs(res_4), norm=LogNorm(),title="Inhomogeneous PSF Response")
    pl.output(name="test_inhomogeneous_2.png")

    # Test Normalization of inhomogeneous psf
    f2 = ift.Field.full(position_space, 1.2)
    f2_s = cut_interpolate_margin(f2)
    res_5 = oa_chandra(f2)
    # np.testing.assert_allclose(res_5.val, f2_s.val)

    ftest = ift.exp(cf)*1e-4 + f2
    ftest_s = cut_interpolate_margin(ftest)
    rtest = oa_chandra(ftest)
    pl = ift.Plot()
    pl.add(ftest_s, norm=LogNorm(),title="full ones")
    pl.add(rtest, norm=LogNorm(), title="conv ones with spvkernel", interpolation="none")
    pl.output(name="test_normalization_test.png")


    extracut = xu.MarginZeroPadder(ift.RGSpace([500,500], distances=position_space.distances), 6).adjoint
    double_test = np.zeros(position_space.shape)
    double_test[256:-256,256:-256] = 1.2
    double_test = ift.Field.from_raw(position_space, double_test)
    double_test_s = cut_interpolate_margin(double_test)
    r_dtest = oa_chandra(double_test)


    pl = ift.Plot()
    pl.add(double_test_s,title="full ones")
    pl.add(r_dtest, title="conv ones with spvkernel",vmin=1.15)
    pl.output(name="test_normalization_test_double.png",interpolation="none")

    pl = ift.Plot()
    pl.add(f2_s, title="full ones")
    pl.add(res_5, title="conv ones with inhomogeneous psf of chandra")
    pl.output(name="test_normalization_chandra_2.png")

    # Test Conservation of Flux
    mock_field = np.zeros([1024, 1024])
    # mock_field[300,300] = 30000
    mock_field[300, 300] = 1
    # mock_field[700,700] = 30000
    mock_field = ift.Field.from_raw(position_space, mock_field)

    cut_mock_field = cut_interpolate_margin(mock_field)
    oa_mock_field = oa_chandra(mock_field)
    test = cut_mock_field-oa_mock_field
    res_6 = cut_mock_field.integrate()
    res_7 = oa_mock_field.integrate()

    # mock_op = zp @ ift.ScalingOperator(position_space, 1)
    # full_op = zp.adjoint(xu.convolve_field_operator(kern, mock_op))
    # conv_fft = full_op(mock_field)
    # res_8 = cut_interpolate_margin(conv_fft).integrate()

    # conv_oa_gauss = convolve_oa(mock_field)
    # res_9 = conv_oa_gauss.integrate()

    pl = ift.Plot()
    pl.add(cut_mock_field, title="Some Points")
    pl.add(oa_mock_field, norm=LogNorm(),title="Convolved with OAChandra")
    # pl.add(conv_fft, title="Convolved with FFT")
    # pl.add(conv_oa_gauss, title="Convolved with OA Gaussian ")
    pl.output(name="test_conservation.png")
    np.testing.assert_allclose(res_6.val, res_7.val)

