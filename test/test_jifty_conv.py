import xubik0 as xu
import nifty8 as ift
import matplotlib.pyplot as plt
import numpy as np
import pytest

def test_nifty_vs_jifty_convolution():
    # set up a space
    sp = ift.RGSpace([30, 30], distances=(0.1, 5))

    # draw random input field f
    f = ift.from_random(sp)

    # some gaussian kernel
    width = 3
    x = y = np.linspace(-width, width, sp.shape[0])
    xv, yv = np.meshgrid(x, y)
    kernel = xu.operators.convolve_utils.gauss(xv, yv, 1)
    kernel = np.fft.fftshift(kernel)
    kernel = ift.makeField(sp, kernel)
    kernel_norm = kernel.integrate().val

    # which is normalized
    proper_kernel = kernel_norm**-1 * kernel

    # Nifty / Xubik Conv result
    so = ift.ScalingOperator(sp, 1)
    conv_op = xu.convolve_field_operator(proper_kernel, so)
    res_1 = conv_op(f)

    # Jifty / Xubik Conv result
    res_2 = xu.jifty_convolve(f.val, proper_kernel.val, sp, axes=(0, 1))

    # Assert
    np.testing.assert_allclose(res_1.val, res_2)
