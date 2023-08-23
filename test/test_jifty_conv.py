import xubik0 as xu
import nifty8 as ift
import matplotlib.pyplot as plt
import numpy as np

# set up a space
sp = ift.RGSpace([30, 30], distances=(0.1, 5))
sp = ift.makeDomain(sp)

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

# Plotting of nifty result
fig, ax = plt.subplots(1, 3)
im = ax[0].imshow(f.val)
fig.colorbar(im, ax=ax[0])
ib = ax[1].imshow(proper_kernel.val)
fig.colorbar(ib, ax=ax[1])
ic = ax[2].imshow(res_1.val)
fig.colorbar(ic, ax=ax[2])
fig.tight_layout()
plt.show()

# consistency check if flux is conserved
print("Integral over the Field", f.integrate().val)
print("Integral over the convolved Field", res_1.integrate().val)

# jubik convolve
res_2 = xu.jifty_convolve(f.val, proper_kernel.val, axes=(0, 1))
cres = res_2*sp[0].scalar_dvol  # FIXME this should be part of the convolution

# plot jubik convolve result without correction
fig, ax = plt.subplots(1, 3)
im = ax[0].imshow(f.val)
fig.colorbar(im, ax=ax[0])
ib = ax[1].imshow(proper_kernel.val)
fig.colorbar(ib, ax=ax[1])
ic = ax[2].imshow(res_2)
fig.colorbar(ic, ax=ax[2])
plt.show()

# plot with correction for scalar dvol
fig, ax = plt.subplots(1, 3)
im = ax[0].imshow(f.val)
fig.colorbar(im, ax=ax[0])
ib = ax[1].imshow(proper_kernel.val)
fig.colorbar(ib, ax=ax[1])
ic = ax[2].imshow(cres)
fig.colorbar(ic, ax=ax[2])
plt.show()

# check if both agree
print("Both methods are the same:", np.allclose(cres, res_1.val))
