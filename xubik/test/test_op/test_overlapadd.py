import nifty8 as ift
import xubik0 as xu
import numpy as np

from PIL import Image

im = np.array(Image.open("Lenna.png"))[:, :, 0]
im = im.astype("float64")
im_roll = np.roll(im, (50, 50))
sp = ift.RGSpace(im.shape)
im_f = ift.Field.from_raw(sp, im)

oa = xu.OverlapAdd(im_f.domain[0], 64, 0)
weights = ift.makeOp(xu.get_weights(oa.target))
zp= xu.MarginZeroPadder(oa.target, 16, space=1)
xu.plot_single_psf(ift.makeField(sp, im_roll), "lenna_roll", logscale=True)
# sr =zp(weights(oa(im_f)))
# back = oa.adjoint(sr)

# pl = ift.Plot()
# pl.add(back)
# pl.output(name="lenna_bf.png")
sp = ift.RGSpace([1024,1024])
domain = ift.makeDomain(sp)
margin = 1
n = 64
kern_domain = ift.makeDomain([ift.UnstructuredDomain(64), sp])
gausskern = xu.get_kernel(200, kern_domain).val

# for i in range(64):
#     tmp_psf = ift.makeField(sp, gausskern.val[i, :, :])
#     gauss_list.append(tmp_psf)

kernels_arr = gausskern
# oa = xu.OverlapAdd(domain[0], n, 0)
# weights = ift.makeOp(xu.get_weights(oa.target))
# zp = xu.MarginZeroPadder(oa.target, margin, space=1)
# padded = zp @ weights @ oa
# cutter = ift.FieldZeroPadder(
#         padded.target, kernels_arr.shape[1:], space=1, central=True
# ).adjoint
# kernels_b = ift.Field.from_raw(cutter.domain, kernels_arr)
# kernels = cutter(kernels_b)
# convolved = xu.convolve_field_operator(kernels, padded, space=1)
# pad_space = ift.RGSpace(
#         [domain.shape[0] + 2 * margin, domain.shape[1] + 2 * margin],
#         distances=domain[0].distances,
# )
# oa_back = xu.OverlapAdd(pad_space, n, margin)
# res = oa_back.adjoint @ convolved
res = xu.OverlapAddConvolver(domain, kernels_arr, n, margin)

star = np.zeros(domain.shape)
star[256,256] = 100
star[20,20] = 100
star[512,512] = 100
star[256,100] = 100
star[100,256] = 100
star = ift.makeField(domain, star)
star_conv = res(star)

pl = ift.Plot()
pl.add(star_conv)
pl.output(name="test.png")

bar_kernel = np.zeros(domain.shape)
for i in range(-15, 15 ,1):
    bar_kernel[0,i] = 20
bar_stack = [bar_kernel for i in range(64)]
bar_stack = np.array(bar_stack)


res2 = xu.OverlapAddConvolver(domain, bar_stack, n, margin)

star_conv = res2(star)

pl = ift.Plot()
pl.add(star_conv)
pl.output(name="test2.png", dpi=600)


hbar_kernel = np.zeros(domain.shape)
for i in range(-15, 15 ,1):
    hbar_kernel[i,i] = 20
hbar_stack = [hbar_kernel for i in range(64)]
hbar_stack = np.array(hbar_stack)
res3 = xu.OverlapAddConvolver(domain, hbar_stack, n, margin)

star_conv = res3(star)
xu.plot_single_psf(ift.makeField(domain,np.roll(hbar_kernel, (488,288), axis=(0,1))), "hbar_kernel_proper", logscale=False)
xu.plot_single_psf(ift.makeField(domain,np.roll(hbar_kernel, 499999 )), "hbar_kernel_inproper", logscale=False)
xu.plot_single_psf(star_conv, "hbar_conv", logscale=True)


psf_file = np.load("../../psf_patches/4952_test.npy", allow_pickle=True)
psfs = []
for p in psf_file:
    psfs.append(p.val)
psfs = np.array(psfs, dtype="float64")
psf_1 = np.roll(psfs[0, :, :], 50000)
psf_2 = np.roll(psf_1,(488,288), axis=(0,1))

max_pos_x, max_pos_y = np.unravel_index(np.argmax(psf_2),(1024,1024))
psf_right = np.roll(psf_2, (-max_pos_x, -max_pos_y), axis=(0,1))
psf_stacks = []
for _ in range(64):
    psf_stacks.append(psf_right)
psf_stacks = np.array(psf_stacks)

res5 = xu.OverlapAddConvolver(domain, psf_stacks, n, margin)
f1 = ift.makeField(domain, psf_1)
f2 = ift.makeField(domain, psf_2)
res4 = xu.OverlapAddConvolver(domain, psfs, n, margin)
star_conv = res4(star)
xu.plot_single_psf(f1, "chandra_psf.png", logscale=True)
xu.plot_single_psf(f2, "chandra_psf_reroll.png", logscale=True)
xu.plot_single_psf(star_conv ,"chandra_conv.png", logscale=True)
star_conv = res5(star)
xu.plot_single_psf(star_conv ,"chandra_conv_right.png", logscale=True)
